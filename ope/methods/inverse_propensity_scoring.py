from typing import Callable, Dict

import pandas as pd

from ..sampling.bootstrap import evaluation_and_bootstrap_metrics


def evaluate(
    df: pd.DataFrame, action_prob_function: Callable, num_bootstrap_samples: int = 0
):
    return evaluation_and_bootstrap_metrics(
        df,
        lambda dataframe: evaluate_raw(dataframe, action_prob_function),
        num_bootstrap_samples=num_bootstrap_samples,
    )


def evaluate_raw(df: pd.DataFrame, action_prob_function: Callable) -> Dict[str, float]:
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
        "expected_reward_new_policy": round(cum_reward_new_policy / len(df), 2),
    }
