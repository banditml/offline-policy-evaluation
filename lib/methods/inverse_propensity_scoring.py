from typing import Callable, Dict

import pandas as pd


def evaluate(df: pd.DataFrame, action_prob_function: Callable) -> Dict[str, float]:
    """
    Inverse propensity scoring (IPS) tutorial:
    https://www.cs.cornell.edu/courses/cs7792/2016fa/lectures/03-counterfactualmodel_6up.pdf
    """

    cum_reward_new_policy = 0
    for _, row in df.iterrows():
        action_probabilities = action_prob_function(row["context"])
        cum_reward_new_policy += (
            action_probabilities[row["action"]] / row["action_prob"]
        ) * row["reward"]

    return {
        "expected_reward_logging_policy": round(df.reward.sum() / len(df), 2),
        "expected_reward_new_policy": cum_reward_new_policy / len(df),
    }
