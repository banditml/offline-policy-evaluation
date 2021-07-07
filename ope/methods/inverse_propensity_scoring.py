from typing import Callable, Dict

import pandas as pd

from ..utils.stats import compute_list_stats


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
) -> Dict[str, Dict[str, float]]:
    """
    Inverse propensity scoring (IPS) tutorial:
    https://www.cs.cornell.edu/courses/cs7792/2016fa/lectures/03-counterfactualmodel_6up.pdf
    """

    results = [
        evaluate_raw(df, action_prob_function, sample=True)
        for _ in range(num_bootstrap_samples)
    ]

    if not results:
        results = [evaluate_raw(df, action_prob_function, sample=False)]

    logging_policy_rewards = [result["logging_policy"] for result in results]
    new_policy_rewards = [result["new_policy"] for result in results]

    return {
        "expected_reward_logging_policy": compute_list_stats(logging_policy_rewards),
        "expected_reward_new_policy": compute_list_stats(new_policy_rewards),
    }


def evaluate_raw(
    df: pd.DataFrame, action_prob_function: Callable, sample: bool
) -> Dict[str, float]:

    tmp_df = df.sample(df.shape[0], replace=True) if sample else df

    cum_reward_new_policy = 0
    for _, row in tmp_df.iterrows():
        action_probabilities = action_prob_function(row["context"])
        cum_reward_new_policy += (
            action_probabilities[row["action"]] / row["action_prob"]
        ) * row["reward"]

    return {
        "logging_policy": tmp_df.reward.sum() / len(tmp_df),
        "new_policy": cum_reward_new_policy / len(tmp_df),
    }
