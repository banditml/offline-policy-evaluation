from typing import Callable, Dict

import numpy as np
import pandas as pd

from ..training.predictor import Predictor
from ..utils.stats import compute_list_stats


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Doubly robust (DR) tutorial:
    https://arxiv.org/pdf/1503.02834.pdf
    """
    # train a model that predicts reward given (context, action)
    reward_model = Predictor()
    reward_model.fit(df)

    results = [
        evaluate_raw(df, action_prob_function, sample=True, reward_model=reward_model)
        for _ in range(num_bootstrap_samples)
    ]

    if not results:
        results = [
            evaluate_raw(
                df, action_prob_function, sample=False, reward_model=reward_model
            )
        ]

    logging_policy_rewards = [result["logging_policy"] for result in results]
    new_policy_rewards = [result["new_policy"] for result in results]

    return {
        "expected_reward_logging_policy": compute_list_stats(logging_policy_rewards),
        "expected_reward_new_policy": compute_list_stats(new_policy_rewards),
    }


def evaluate_raw(
    df: pd.DataFrame,
    action_prob_function: Callable,
    sample: bool,
    reward_model: Predictor,
) -> Dict[str, float]:

    tmp_df = df.sample(df.shape[0], replace=True) if sample else df

    context_df = tmp_df.context.apply(pd.Series)
    context_array = context_df[reward_model.context_column_order].values
    cum_reward_new_policy = 0

    for idx, row in tmp_df.iterrows():
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
        "logging_policy": tmp_df.reward.sum() / len(tmp_df),
        "new_policy": cum_reward_new_policy / len(tmp_df),
    }
