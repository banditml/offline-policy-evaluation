from collections import defaultdict
import json
import time

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
import torch

from ..models import embed_dnn
from ..preprocessing import preprocessor


class BanditPredictor:
    """Class used to make predictions given a trained bandit model."""

    def __init__(
        self,
        experiment_params,
        float_feature_order,
        id_feature_order,
        id_feature_str_to_int_map,
        transforms,
        imputers,
        model,
        reward_type,
        model_spec=None,
    ):
        self.experiment_params = experiment_params
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.id_feature_str_to_int_map = id_feature_str_to_int_map
        self.transforms = transforms
        self.imputers = imputers
        self.model = model
        self.reward_type = reward_type
        self.model_spec = model_spec

        # the ordered decisions that we need to score over.
        self.decisions = []

        # if using dropout to get prediction uncertainty, how many times to score
        # the same observation
        self.num_times_to_score = 30
        self.ucb_percentile = 90

    def transform_feature(self, vals, transformer=None, imputer=None):
        vals = vals.reshape(-1, 1)
        if imputer:
            vals = imputer.transform(vals)
        if transformer:
            vals = transformer.transform(vals)

        return vals

    def preprocess_input(self, input):
        # score all decisions so expand the input across decisions
        decision_meta = self.experiment_params["features"]["decision"]
        if decision_meta["type"] == "C":
            decisions = decision_meta["possible_values"]
            expanded_input = [dict(input, **{"decision": d}) for d in decisions]
        elif decision_meta["type"] == "P":
            product_set_id = decision_meta["product_set_id"]
            product_set = self.experiment_params["product_sets"][product_set_id]
            decisions = product_set["ids"]
            expanded_input = [dict(input, **{"decision": d}) for d in decisions]

        self.decisions = decisions
        df = pd.DataFrame(expanded_input)
        float_feature_array = np.empty((len(df), 0))
        id_list_feature_array = np.empty((len(df), 0))

        for feature_name in self.float_feature_order:
            values = self.transform_feature(
                df[feature_name].values,
                self.transforms[feature_name],
                self.imputers[feature_name],
            )
            float_feature_array = np.append(float_feature_array, values, axis=1)

        for feature_name in self.id_feature_order:
            meta = self.experiment_params["features"][feature_name]
            product_set_id = meta["product_set_id"]
            product_set_meta = self.experiment_params["product_sets"][product_set_id]
            if meta["use_dense"] is True and "dense" in product_set_meta:

                dense = defaultdict(list)
                # TODO: don't like that this is O(n^2), think about better way to do this
                for val in df[feature_name].values:
                    for idx, feature_spec in enumerate(product_set_meta["features"]):
                        dense_feature_name = feature_spec["name"]
                        dense_feature_val = product_set_meta["dense"][val][idx]
                        dense[dense_feature_name].append(dense_feature_val)

                for idx, feature_spec in enumerate(product_set_meta["features"]):
                    dtype = (
                        np.dtype(float)
                        if feature_spec["type"] == "N"
                        else np.dtype(object)
                    )
                    vals = np.array(dense[feature_spec["name"]], dtype=dtype)
                    values = self.transform_feature(
                        vals,
                        self.transforms[feature_spec["name"]],
                        self.imputers[feature_spec["name"]],
                    )
                    float_feature_array = np.append(float_feature_array, values, axis=1)
            else:
                # sparse id list features need to be converted from string to int,
                # but aside from that are not imputed or transformed.
                str_to_int_map = self.id_feature_str_to_int_map[product_set_id]
                values = self.transform_feature(
                    df[feature_name].apply(lambda x: str_to_int_map[x]).values
                )
                id_list_feature_array = np.append(id_list_feature_array, values, axis=1)

        return {
            "X_float": pd.DataFrame(float_feature_array),
            "X_id_list": pd.DataFrame(id_list_feature_array),
        }

    def preprocessed_input_to_pytorch(self, data):
        X, _ = preprocessor.data_to_pytorch(data)
        return X

    def predict(self, input, get_ucb_scores=False):
        """
        If `get_ucb_scores` is True, get upper confidence bound scores which
        requires a model trained with dropout and for the model to be in train()
        mode (eval model turns off dropout by default).
        """
        input = self.preprocess_input(input)
        pytorch_input = self.preprocessed_input_to_pytorch(input)

        with torch.no_grad():
            scores = self.model.forward(**pytorch_input)

            # for binary classification we just need the score for the `1` label
            if self.reward_type == "binary":
                scores = scores[:, 1:]

            ucb_scores = []

            if get_ucb_scores:
                assert (
                    self.model.use_dropout is True
                ), "Can only get UCB scores if model was trained with dropout."
                self.model.train()
                scores_samples = torch.tensor(
                    [
                        self.model.forward(**pytorch_input).numpy()
                        for i in range(self.num_times_to_score)
                    ]
                )
                ucb_scores = np.percentile(scores_samples, q=95, axis=0).tolist()

        return {
            "scores": scores.tolist(),
            "ids": self.decisions,
            "ucb_scores": ucb_scores,
        }

    def model_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method
        writes the PyTorch net to file using PyTorch's built in serialization.
        """
        torch.save(self.model.state_dict(), path)

    def config_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method writes
        the parameters to reconstruct the preprocessing & data imputation
        objects/logic to a JSON file.
        """
        output = {
            "model_spec": self.model_spec,
            "experiment_params": self.experiment_params,
            "float_feature_order": self.float_feature_order,
            "id_feature_order": self.id_feature_order,
            "id_feature_str_to_int_map": self.id_feature_str_to_int_map,
            "reward_type": self.reward_type,
            "transforms": {},
            "imputers": {},
        }

        # write the parameters of the feature transformers
        for feature_name, transform in self.transforms.items():
            if transform is None:
                # id lists don't have transforms
                spec = None
            elif isinstance(transform, preprocessing.StandardScaler):
                spec = {
                    "name": "StandardScaler",
                    "mean": transform.mean_.tolist(),
                    "var": transform.var_.tolist(),
                    "scale": transform.scale_.tolist(),
                }
            elif isinstance(transform, preprocessing.OneHotEncoder):
                spec = {
                    "name": "OneHotEncoder",
                    "categories": [transform.categories_[0].tolist()],
                    "sparse": transform.sparse,
                }
            else:
                raise Exception(
                    f"Don't know how to serialize preprocessor of type {type(transform)}"
                )
            output["transforms"][feature_name] = spec

        # write the parameters of the feature imputers
        for feature_name, imputer in self.imputers.items():
            if imputer is None:
                # categorical & id lists don't have imputers
                spec = None
            else:
                spec = {
                    "parameters": imputer.get_params(),
                    "statistics": imputer.statistics_.tolist(),
                }
            output["imputers"][feature_name] = spec

        with open(path, "w") as f:
            json.dump(output, f)

    @staticmethod
    def predictor_from_file(config_path, net_path):
        with open(config_path, "rb") as f:
            config_dict = json.load(f)

        # initialize the pytorch model and put it in `eval` mode
        net = embed_dnn.EmbedDnn(**config_dict["model_spec"])
        net.load_state_dict(torch.load(net_path))
        net.eval()

        # initialize transforms
        transforms = {}
        for feature_name, transform_spec in config_dict["transforms"].items():
            if transform_spec is None:
                # id lists don't have transforms
                transform = None
            elif transform_spec["name"] == "StandardScaler":
                transform = preprocessing.StandardScaler()
                transform.mean_ = np.array(transform_spec["mean"])
                transform.scale_ = np.array(transform_spec["scale"])
                transform.var_ = np.array(transform_spec["var"])
            elif transform_spec["name"] == "OneHotEncoder":
                transform = preprocessing.OneHotEncoder()
                transform.sparse = transform_spec["sparse"]
                transform.categories_ = np.array(transform_spec["categories"])
            else:
                raise Exception(
                    f"Don't know how to load transform_spec of type {transform_spec['name']}"
                )
            transforms[feature_name] = transform

        # initialize imputers
        imputers = {}
        for feature_name, imputer_spec in config_dict["imputers"].items():
            if imputer_spec is None:
                # categoricals & id lists don't have imputers
                imputer = None
            else:
                imputer = SimpleImputer()
                imputer.set_params(**imputer_spec["parameters"])
                imputer.statistics_ = np.array(imputer_spec["statistics"])
            imputers[feature_name] = imputer

        return BanditPredictor(
            experiment_params=config_dict["experiment_params"],
            float_feature_order=config_dict["float_feature_order"],
            id_feature_order=config_dict["id_feature_order"],
            id_feature_str_to_int_map=config_dict["id_feature_str_to_int_map"],
            transforms=transforms,
            imputers=imputers,
            model=net,
            reward_type=config_dict["reward_type"],
            model_spec=config_dict["model_spec"],
        )
