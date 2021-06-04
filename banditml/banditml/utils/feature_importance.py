from typing import Tuple

import torch

from . import model_constructors, utils

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


def keep_top_n_features(n, X, feature_order, feature_scores):
    """Keep top n features based on feature importance."""

    top_n_feature_names = []
    top_n_feature_idxs = []
    for score, feature_name in feature_scores[:n]:
        top_n_feature_names.append(feature_name)
        top_n_feature_idxs.append(feature_order.index(feature_name))

    idxs = torch.tensor(top_n_feature_idxs)
    X_float = torch.index_select(X["X_float"], 1, idxs)
    X["X_float"] = X_float
    return X, top_n_feature_names
