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
        experiment_specific_params,
        float_feature_order,
        id_feature_order,
        transforms,
        imputers,
        net,
        net_spec=None,
    ):
        self.experiment_specific_params = experiment_specific_params
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.transforms = transforms
        self.imputers = imputers
        self.net = net
        self.net_spec = net_spec

        # the ordered decisions that we need to score over.
        self.decisions = []

    def transform_feature(self, vals, transformer=None, imputer=None):
        vals = vals.reshape(-1, 1)
        if imputer:
            vals = imputer.transform(vals)
        if transformer:
            vals = transformer.transform(vals)

        return vals

    def preprocess_input(self, input):
        # score all decisions so expand the input across decisions
        decision_meta = self.experiment_specific_params["features"]["decision"]
        if decision_meta["type"] == "C":
            decisions = decision_meta["possible_values"]
            expanded_input = [dict(input, **{"decision": d}) for d in decisions]
        elif decision_meta["type"] == "P":
            product_set_id = decision_meta["product_set_id"]
            product_set = self.experiment_specific_params["product_sets"][
                product_set_id
            ]
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
            meta = self.experiment_specific_params["features"][feature_name]
            product_set_id = meta["product_set_id"]
            product_set_meta = self.experiment_specific_params["product_sets"][
                product_set_id
            ]
            if meta["use_dense"] is True and "dense" in product_set_meta:
                # if dense is true then convert ID's into their dense features
                dense = np.array(
                    [product_set_meta["dense"][str(i)] for i in df[feature_name].values]
                )
                for idx, feature in enumerate(product_set_meta["features"]):
                    vals = dense[:, idx]
                    values = self.transform_feature(
                        vals,
                        self.transforms[feature["name"]],
                        self.imputers[feature["name"]],
                    )
                    float_feature_array = np.append(float_feature_array, values, axis=1)
            else:
                # sparse id list features aren't preprocessed, instead they use an
                # embedding table which is built into the pytorch model
                values = self.transform_feature(df[feature_name].values)
                id_list_feature_array = np.append(id_list_feature_array, values, axis=1)

        return {
            "X_float": pd.DataFrame(float_feature_array),
            "X_id_list": pd.DataFrame(id_list_feature_array),
        }

    def preprocessed_input_to_pytorch(self, data):
        X, _ = preprocessor.data_to_pytorch(data)
        return X

    def predict(self, input):
        input = self.preprocess_input(input)
        pytorch_input = self.preprocessed_input_to_pytorch(input)
        scores = self.net.forward(**pytorch_input)
        return {"scores": scores.tolist(), "ids": self.decisions}

    def net_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method
        writes the PyTorch net to file using PyTorch's built in serialization.
        """
        torch.save(self.net.state_dict(), path)

    def config_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method writes
        the parameters to reconstruct the preprocessing & data imputation
        objects/logic to a JSON file.
        """
        output = {
            "net_spec": self.net_spec,
            "experiment_specific_params": self.experiment_specific_params,
            "float_feature_order": self.float_feature_order,
            "id_feature_order": self.id_feature_order,
            "transforms": {},
            "imputers": {},
        }

        # write the parameters of the feature transformers
        for feature_name, transform in self.transforms.items():
            if isinstance(transform, preprocessing.StandardScaler):
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
            elif isinstance(transform, str):
                # id lists don't have transforms
                spec = None
            else:
                raise Exception(
                    f"Don't know how to serialize preprocessor of type {type(transform)}"
                )
            output["transforms"][feature_name] = spec

        # write the parameters of the feature imputers
        for feature_name, imputer in self.imputers.items():
            if isinstance(imputer, str):
                # id lists don't have imputers
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
        net = embed_dnn.EmbedDnn(**config_dict["net_spec"])
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
                # id lists don't have imputers
                imputer = None
            else:
                imputer = SimpleImputer()
                imputer.set_params(**imputer_spec["parameters"])
                imputer.statistics_ = np.array(imputer_spec["statistics"])
            imputers[feature_name] = imputer

        return BanditPredictor(
            experiment_specific_params=config_dict["experiment_specific_params"],
            float_feature_order=config_dict["float_feature_order"],
            id_feature_order=config_dict["id_feature_order"],
            transforms=transforms,
            imputers=imputers,
            net=net,
            net_spec=config_dict["net_spec"],
        )
