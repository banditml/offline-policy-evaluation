from typing import NoReturn, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from ..estimators import linear, gbdt


class Predictor:
    def _preprocess_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        # preprocess context
        context_df = df.context.apply(pd.Series)
        self.context_column_order = list(context_df.columns)

        # preprocess actions
        self.action_preprocessor = OneHotEncoder(sparse=False)
        action_values = df.action.values.reshape(-1, 1)
        self.possible_actions = set(action_values.squeeze().tolist())
        one_hot_action_values = self.action_preprocessor.fit_transform(action_values)

        X_train = np.concatenate((context_df.values, one_hot_action_values), axis=1)
        y_train = df.reward.values

        return X_train, y_train

    def fit(self, df: pd.DataFrame) -> NoReturn:
        X_train, y_train = self._preprocess_data(df)
        results = gbdt.fit_gbdt_regression(X_train, y_train)
        self.model = results.pop("model")
        self.training_stats = results

    def predict(self, input: np.ndarray) -> float:
        return self.model.predict(input)[0]
