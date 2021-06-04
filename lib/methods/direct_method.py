from typing import Callable, Dict

import numpy as np
import pandas as pd

from ..training.predictor import Predictor


def evaluate(df: pd.DataFrame, action_prob_function: Callable) -> Dict[str, float]:
    """
    Direct method (DM) tutorial:
    See section 3.2 in https://arxiv.org/pdf/1503.02834.pdf
    """

    # train a model that predicts reward given (context, action)
    reward_model = Predictor()
    reward_model.fit(df)

    context_df = df.context.apply(pd.Series)
    context_array = context_df[reward_model.context_column_order].values
    cum_reward_new_policy = 0

    for idx, row in df.iterrows():
        observation_expected_reward = 0
        action_probabilities = action_prob_function(row["context"])
        for action, action_probability in action_probabilities.items():
            one_hot_action = reward_model.action_preprocessor.transform(
                np.array(action).reshape(-1, 1)
            )
            observation = np.concatenate((context_array[idx], one_hot_action.squeeze()))
            predicted_reward = reward_model.predict(observation.reshape(1, -1))
            observation_expected_reward += action_probability * predicted_reward
        cum_reward_new_policy += observation_expected_reward

    return {
        "expected_reward_logging_policy": round(df.reward.sum() / len(df), 2),
        "expected_reward_new_policy": round(cum_reward_new_policy / len(df), 2),
    }
