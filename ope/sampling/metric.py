from typing import Callable, Dict, Union

import pandas as pd


def confidence_interval(p_value: float) -> Callable:
    """Returns a callable which evaluates a confidence interval on a series."""

    def get_confidence_interval(series: pd.Series) -> Dict[str, Union[str, float]]:
        mean = series.mean()
        standard_dev = series.std()
        return {
            "name": f"confidence_interval_{p_value}",
            "value": [mean - 1.96 * standard_dev, mean + 1.96 * standard_dev],
        }

    return get_confidence_interval
