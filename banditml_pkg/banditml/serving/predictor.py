import time

import numpy as np
import pandas as pd
import torch

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
    ):
        self.experiment_specific_params = experiment_specific_params
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.transforms = transforms
        self.imputers = imputers
        self.net = net

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
        return self.net.predict(pytorch_input)
