import json
import logging
import pickle
import time
from collections import defaultdict

import numpy as np
import pandas as pd

import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer

from ..models import embed_dnn
from ..preprocessing import preprocessor

logger = logging.getLogger(__name__)


class BanditPredictor:
    """Class used to make predictions given a trained bandit model."""

    def __init__(
        self,
        feature_config,
        float_feature_order,
        id_feature_order,
        id_feature_str_to_int_map,
        transforms,
        imputers,
        model,
        model_type,
        reward_type,
        model_spec=None,
        dense_features_to_use=["*"],
    ):
        self.feature_config = feature_config
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.id_feature_str_to_int_map = id_feature_str_to_int_map
        self.transforms = transforms
        self.imputers = imputers
        self.model = model
        self.model_type = model_type
        self.reward_type = reward_type
        self.model_spec = model_spec
        self.dense_features_to_use = dense_features_to_use

        # the ordered choices that we need to score over.
        self.choices = feature_config["choices"]

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

    def preprocess_input(self, input, choices):
        # score input choices or all choices by default if none provided.
        # expand the input across all of these choices
        self.scored_choices = choices = choices or self.choices
        expanded_input = [
            dict(input, **{preprocessor.DECISION_FEATURE_NAME: d})
            for d in self.scored_choices
        ]

        df = pd.DataFrame(expanded_input)
        float_feature_array = np.empty((len(df), 0))
        id_list_feature_array = np.empty((len(df), 0))

        for feature_name in self.float_feature_order:
            if feature_name not in df.columns:

                if feature_name == preprocessor.POSITION_FEATURE_NAME:
                    # always score position feature at a fixed position
                    # since including it in training was just meant to debias
                    df[feature_name] = 0

                else:
                    # context is missing this feature, that is fine
                    logger.warning(
                        f"'{feature_name}' expected in context, but missing."
                    )
                    if self.feature_config["features"][feature_name]["type"] == "C":
                        df[feature_name] = preprocessor.MISSING_CATEGORICAL_CATEGORY
                    else:
                        df[feature_name] = None

            values = self.transform_feature(
                df[feature_name].values,
                self.transforms[feature_name],
                self.imputers[feature_name],
            )
            float_feature_array = np.append(float_feature_array, values, axis=1)

        for feature_name in self.id_feature_order:
            meta = self.feature_config["features"][feature_name]
            product_set_id = meta["product_set_id"]
            product_set_meta = self.feature_config["product_sets"][product_set_id]

            if feature_name not in df.columns:
                # Handle passing missing product set feature
                logger.warning(f"'{feature_name}' expected in context, but missing.")
                df[feature_name] = None

            if meta["use_dense"] is True and "dense" in product_set_meta:

                dense = defaultdict(list)
                # TODO: don't like that this is O(n^2), think about better way to do this
                for val in df[feature_name].values:
                    if not isinstance(val, list):
                        val = [val]

                    dense_matrix = []
                    for v in val:
                        dense_features = product_set_meta["dense"].get(v)
                        if not dense_features:
                            logger.warning(
                                f"No dense representation found for '{feature_name}'"
                                f" product set value '{v}'."
                            )
                        else:
                            dense_matrix.append(dense_features)

                    if not dense_matrix:
                        # there were no or no valid id features to add, add an
                        # empty row to be imputed
                        dense_matrix.append([])

                    for idx, feature_spec in enumerate(product_set_meta["features"]):
                        dense_feature_name = feature_spec["name"]
                        row_vals = []
                        for row in dense_matrix:
                            if not row:
                                dense_feature_val = (
                                    preprocessor.MISSING_CATEGORICAL_CATEGORY
                                    if feature_spec["type"] == "C"
                                    else None
                                )
                            else:
                                dense_feature_val = row[idx]
                            row_vals.append(dense_feature_val)

                        dense_feature_val = preprocessor.flatten_dense_id_list_feature(
                            row_vals, feature_spec["type"]
                        )
                        dense[dense_feature_name].append(dense_feature_val)

                for idx, feature_spec in enumerate(product_set_meta["features"]):
                    dense_feature_name_desc = f"{feature_name}:{feature_spec['name']}"
                    if (
                        self.dense_features_to_use != ["*"]
                        and dense_feature_name_desc not in self.dense_features_to_use
                    ):
                        continue

                    dtype = (
                        np.dtype(float)
                        if feature_spec["type"] == "N"
                        else np.dtype(object)
                    )

                    vals = dense[feature_spec["name"]]
                    if feature_spec["type"] == "C":
                        # fill in null categorical values with a "null" category
                        vals = [
                            preprocessor.MISSING_CATEGORICAL_CATEGORY
                            if v is None
                            else v
                            for v in vals
                        ]

                    vals = np.array(vals, dtype=dtype)
                    values = self.transform_feature(
                        vals,
                        self.transforms[dense_feature_name_desc],
                        self.imputers[dense_feature_name_desc],
                    )
                    float_feature_array = np.append(float_feature_array, values, axis=1)
            else:
                # sparse id list features need to be converted from string to int,
                # but aside from that are not imputed or transformed.
                str_to_int_map = self.id_feature_str_to_int_map[product_set_id]
                # if the feature value is not present in the map, assign it to 0
                # which corresponds to the null embedding row
                values = self.transform_feature(
                    df[feature_name].apply(lambda x: str_to_int_map.get(x, 0)).values
                )
                id_list_feature_array = np.append(id_list_feature_array, values, axis=1)

        return {
            "X_float": pd.DataFrame(float_feature_array),
            "X_id_list": pd.DataFrame(id_list_feature_array),
        }

    def preprocessed_input_to_pytorch(self, data):
        X, _ = preprocessor.data_to_pytorch(data)
        return X

    def predict(self, input, choices=None, get_ucb_scores=False):
        """
        If `get_ucb_scores` is True, get upper confidence bound scores which
        requires a model trained with dropout and for the model to be in train()
        mode (eval model turns off dropout by default).
        """
        input = self.preprocess_input(input, choices)
        pytorch_input = self.preprocessed_input_to_pytorch(input)

        ucb_scores = None
        if self.model_type == "neural_bandit":
            # pytorch model
            with torch.no_grad():
                scores = self.model.forward(**pytorch_input)

                # for binary classification we just need the score for the `1` label
                if self.reward_type == "binary":
                    scores = scores[:, 1:]

                if get_ucb_scores:

                    assert (
                        self.model.use_dropout is True or self.model.is_mdn is True
                    ), "Can only get UCB scores if model was trained with dropout or MDN."

                    # Currently this assumes you only picked one or the other...
                    if self.model.use_dropout:
                        self.model.train()
                        scores_samples = torch.tensor(
                            [
                                self.model.forward(**pytorch_input).numpy()
                                for i in range(self.num_times_to_score)
                            ]
                        )

                        ucb_scores = np.percentile(
                            scores_samples, q=95, axis=0
                        ).tolist()

                    else:
                        batch_size = self.model.batch_size
                        idx = range(scores_samples.shape[0])
                        mu_est = [i for i in idx if i // batch_size % 2 == 0]
                        var_est = [i for i in idx if i // batch_size % 2 == 1]
                        ucb_scores = (
                            scores_samples[mu_est] + 1.96 * scores_samples[var_est]
                        )

        elif self.model_type in (
            "linear_bandit",
            "gbdt_bandit",
            "random_forest_bandit",
        ):
            if self.reward_type == "binary":
                scores = self.model.predict_proba(pytorch_input["X_float"])
                scores = scores[:, 1:]
            else:
                scores = self.model.predict(pytorch_input["X_float"])
        else:
            raise Exception(
                f"predict() for model type {self.model_type} not supported."
            )

        if not ucb_scores:
            ucb_scores = [0.0 for i in range(len(scores))]

        # convert scores to list of floats and round
        scores = scores.squeeze().tolist()
        scores = [scores] if not isinstance(scores, list) else scores
        scores = [round(score, 4) for score in scores]

        # sort by scores before returning for visual convenience
        zipped = zip(scores, self.scored_choices, ucb_scores)
        sorted_scores = sorted(zipped, reverse=True)
        scores, ids, ucb_scores = zip(*sorted_scores)

        return {
            "scores": list(scores),
            "ids": list(ids),
            "ucb_scores": list(ucb_scores),
        }

    def model_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method
        writes the model to file using PyTorch's built in serialization (which
        uses pickle) or explicitly using pickle for sklearn models.
        """
        # TODO: stop using pickle
        if self.model_type == "neural_bandit":
            # https://pytorch.org/tutorials/beginner/saving_loading_models.html#what-is-a-state-dict
            # torch.save: Saves a serialized object to disk. This function uses
            # Python’s pickle utility for serialization. Models, tensors, and
            # dictionaries of all kinds of objects can be saved using this function.
            torch.save(self.model.state_dict(), path)
        else:
            # https://scikit-learn.org/stable/modules/model_persistence.html
            with open(path, "wb") as f:
                pickle.dump(self.model, f)

    def config_to_file(self, path):
        """
        Writing a Predictor object to file requires two files. This method writes
        the parameters to reconstruct the preprocessing & data imputation
        objects/logic to a JSON file.
        """
        output = {
            "model_type": self.model_type,
            "model_spec": self.model_spec,
            "feature_config": self.feature_config,
            "float_feature_order": self.float_feature_order,
            "id_feature_order": self.id_feature_order,
            "id_feature_str_to_int_map": self.id_feature_str_to_int_map,
            "dense_features_to_use": self.dense_features_to_use,
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
    def predictor_from_file(config_path, model_path):
        with open(config_path, "rb") as f:
            config_dict = json.load(f)

        if config_dict["model_type"] == "neural_bandit":
            # initialize the pytorch model and put it in `eval` mode
            model = embed_dnn.EmbedDnn(**config_dict["model_spec"])
            model.load_state_dict(torch.load(model_path))
            model.eval()
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

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
            feature_config=config_dict["feature_config"],
            float_feature_order=config_dict["float_feature_order"],
            id_feature_order=config_dict["id_feature_order"],
            id_feature_str_to_int_map=config_dict["id_feature_str_to_int_map"],
            transforms=transforms,
            imputers=imputers,
            model=model,
            model_type=config_dict["model_type"],
            reward_type=config_dict["reward_type"],
            model_spec=config_dict["model_spec"],
            dense_features_to_use=config_dict["dense_features_to_use"],
        )
