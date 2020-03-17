import time

import numpy as np
import pandas as pd
import torch

from ml.preprocessing import preprocessor


class BanditPredictor:
    """Class used to make predictions given a trained bandit model."""

    def __init__(
        self,
        experiment_specific_params,
        float_feature_order,
        id_feature_order,
        transforms,
        net,
    ):
        self.experiment_specific_params = experiment_specific_params
        self.float_feature_order = float_feature_order
        self.id_feature_order = id_feature_order
        self.transforms = transforms
        self.net = net

    def preprocess_input(self, input):
        # score all decisions so expand the input across decisions
        decision_meta = self.experiment_specific_params["features"]["decision"]
        if decision_meta["type"] == "C":
            decisions = decision_meta["possible_values"]
            expanded_input = [dict(input, **{"decision": d}) for d in decisions]
        elif decision_meta["type"] == "P":
            pass

        df = pd.DataFrame(expanded_input)
        float_feature_array = np.empty((len(df), 0))
        id_list_feature_array = np.empty((len(df), 0))

        for feature_name in self.float_feature_order:
            transformer = self.transforms[feature_name]
            values = transformer.transform(df[feature_name].values.reshape(-1, 1))
            float_feature_array = np.append(float_feature_array, values, axis=1)

        for feature_name in self.id_feature_order:
            # sparse id list features aren't preprocessed, instead they use an
            # embedding table which is built into the pytorch model
            values = df[feature_name].values.reshape(-1, 1)
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
