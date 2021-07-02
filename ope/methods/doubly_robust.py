from typing import Callable, Dict, Union

import numpy as np
import pandas as pd

from ..sampling.bootstrap import evaluation_and_bootstrap_metrics
from ..training.predictor import Predictor


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Union[Dict, float]]:
    """
    Doubly robust (DR) tutorial:
    https://arxiv.org/pdf/1503.02834.pdf
    """
    # train a model that predicts reward given (context, action)
    reward_model = Predictor()
    reward_model.fit(df)

    return evaluation_and_bootstrap_metrics(
        df,
        lambda data: evaluate_with_model(data, action_prob_function, reward_model),
        num_bootstrap_samples=num_bootstrap_samples,
    )


def evaluate_with_model(
    df: pd.DataFrame, action_prob_function: Callable, reward_model: Predictor
) -> Dict[str, float]:
    """Evaluate a policy on a dataset with a pre-trained model"""
    context_df = df.context.apply(pd.Series)
    context_array = context_df[reward_model.context_column_order].values
    cum_reward_new_policy = 0

    for idx, row in df.iterrows():
        observation_expected_reward = 0
        processed_context = context_array[idx]

        # first compute the left hand term, which is the direct method
        action_probabilities = action_prob_function(row["context"])
        for action, action_probability in action_probabilities.items():
            one_hot_action = reward_model.action_preprocessor.transform(
                np.array(action).reshape(-1, 1)
            )
            observation = np.concatenate((processed_context, one_hot_action.squeeze()))
            predicted_reward = reward_model.predict(observation.reshape(1, -1))
            observation_expected_reward += action_probability * predicted_reward

        # then compute the right hand term, which is similar to IPS
        logged_action = row["action"]
        new_action_probability = action_probabilities[logged_action]
        weight = new_action_probability / row["action_prob"]
        one_hot_action = reward_model.action_preprocessor.transform(
            np.array(row["action"]).reshape(-1, 1)
        )
        observation = np.concatenate((processed_context, one_hot_action.squeeze()))
        predicted_reward = reward_model.predict(observation.reshape(1, -1))
        observation_expected_reward += weight * (row["reward"] - predicted_reward)

        cum_reward_new_policy += observation_expected_reward

    return {
        "expected_reward_logging_policy": round(df.reward.sum() / len(df), 2),
        "expected_reward_new_policy": round(cum_reward_new_policy / len(df), 2),
    }
