from typing import List
import statistics

P95_Z_SCORE = 1.96


def compute_list_stats(input: List):
    """Compute mean and P95 CI of mean for a list of floats."""
    mean = statistics.mean(input)
    std_dev = statistics.stdev(input) if len(input) > 1 else None
    ci_low = round(mean - P95_Z_SCORE * std_dev, 2) if std_dev else None
    ci_high = round(mean + P95_Z_SCORE * std_dev, 2) if std_dev else None

    return {"mean": round(mean, 2), "ci_low": ci_low, "ci_high": ci_high}
