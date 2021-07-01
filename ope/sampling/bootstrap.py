from typing import Callable, Iterable, Dict

import pandas as pd
from pandas import DataFrame

from .metric import confidence_interval


def evaluation_and_bootstrap_metrics(
    data: DataFrame,
    evaluator: Callable[[DataFrame], Dict[str, Dict]],
    metrics: Iterable[Callable] = [confidence_interval(0.95)],
    num_bootstrap_samples: int = 50,
):
    evaluation = evaluator(data)

    if num_bootstrap_samples:
        bootstrap_evaluation = bootstrap_metrics(
            data, evaluator, metrics=metrics, num_samples=num_bootstrap_samples
        )
        evaluation = {key: {'value': value} for key, value in
                      evaluation.items()}

        for key in evaluation:
            evaluation[key].update(bootstrap_evaluation[key])

    return evaluation


def bootstrap_metrics(
    data: DataFrame,
    evaluator: Callable[[DataFrame], Dict[str, Dict]],
    metrics: Iterable[Callable] = [confidence_interval(0.95)],
    num_samples: int = 50,
):

    outcomes = [evaluator(sample) for sample in bootstrap_samples(data, num_samples)]

    outcomes_df = pd.DataFrame(outcomes)
    outputs = {}

    for col_name, values in outcomes_df.iteritems():
        column_metrics = {}

        for metric_function in metrics:
            metric = metric_function(values)
            column_metrics[metric["name"]] = metric["value"]

        outputs[col_name] = column_metrics

    return outputs


def bootstrap_samples(dataframe: DataFrame, num_samples: int) -> Iterable[DataFrame]:
    num_rows = dataframe.shape[0]
    for _ in range(num_samples):
        yield dataframe.sample(num_rows, replace=True)
