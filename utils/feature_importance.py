from typing import Tuple

from utils import utils, model_constructors

logger = utils.get_logger(__name__)


def calculate_feature_importance(reward_type, feature_names, X, y):
    """Uses random forest to calculate feature importance."""

    X = X["X_float"]

    model = model_constructors.build_random_forest(reward_type)
    model.fit(X, y.squeeze())

    zipped_importances = zip(model.feature_importances_, feature_names)
    return sorted(zipped_importances, reverse=True)


def display_feature_importances(scores: Tuple[float, str]):
    for score, feature_name in scores:
        logger.info(f"{feature_name}: {score}")
