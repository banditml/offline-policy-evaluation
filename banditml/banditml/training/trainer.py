import argparse
import json
import os
import shutil
import time
from typing import Dict

import pandas as pd

from ..preprocessing import preprocessor
from ..serving.predictor import BanditPredictor
from ..utils import feature_importance, model_constructors, model_trainers, utils

logger = utils.get_logger(__name__)


def num_float_dim(data):
    return len(data["X_float"].columns)


def train(
    training_df: pd.DataFrame,
    ml_config: Dict,
    feature_config: Dict,
    predictor_save_dir: str = None,
) -> BanditPredictor:

    start = time.time()

    utils.validate_ml_config(ml_config)

    logger.info("Checking schema of training data...")
    utils.validate_training_data_schema(training_df)

    if len(training_df) == 0:
        logger.error(f"Got no raws of training data. Training aborted.")
        return None

    logger.info(f"Got {len(training_df)} rows of training data.")
    logger.info(training_df.head())

    utils.fancy_print("Kicking off data preprocessing")

    # always add decision as a feature to use if not using all features
    features_to_use = ml_config["features"].get("features_to_use", ["*"])
    if features_to_use != ["*"]:
        features_to_use.append(preprocessor.DECISION_FEATURE_NAME)
    features_to_use = list(set(features_to_use))
    dense_features_to_use = ml_config["features"].get("dense_features_to_use", ["*"])

    data = preprocessor.preprocess_data(
        training_df,
        feature_config,
        ml_config["reward_type"],
        features_to_use,
        dense_features_to_use,
    )
    X, y = preprocessor.data_to_pytorch(data)

    model_type = ml_config["model_type"]
    model_params = ml_config["model_params"][model_type]
    reward_type = ml_config["reward_type"]

    feature_importance_params = ml_config.get("feature_importance", {})
    if feature_importance_params.get("calc_feature_importance", False):
        # calculate feature importances - only works on non id list features at this time
        utils.fancy_print("Calculating feature importances")
        feature_scores = feature_importance.calculate_feature_importance(
            reward_type=reward_type,
            feature_names=data["final_float_feature_order"],
            X=X,
            y=y,
        )
        feature_importance.display_feature_importances(feature_scores)

        # TODO: Make keeping the top "n" features work in predictor. Right now
        # using this feature breaks predictor, so don't use it in a final model,
        # just use it to experiment in seeing how model performance is.
        if feature_importance_params.get("keep_only_top_n", False):
            utils.fancy_print("Keeping only top N features")
            X, final_float_feature_order = feature_importance.keep_top_n_features(
                n=feature_importance_params["n"],
                X=X,
                feature_order=data["final_float_feature_order"],
                feature_scores=feature_scores,
            )
            data["final_float_feature_order"] = final_float_feature_order
            logger.info(f"Keeping top {feature_importance_params['n']} features:")
            logger.info(final_float_feature_order)

    utils.fancy_print("Starting training")
    # build the model
    if model_type == "neural_bandit":
        model_spec, model = model_constructors.build_pytorch_net(
            feature_specs=feature_config["features"],
            product_sets=feature_config["product_sets"],
            float_feature_order=data["final_float_feature_order"],
            id_feature_order=data["final_id_feature_order"],
            reward_type=reward_type,
            layers=model_params["layers"],
            activations=model_params["activations"],
            dropout_ratio=model_params["dropout_ratio"],
            input_dim=num_float_dim(data),
        )
        logger.info(f"Initialized model: {model}")
    elif model_type == "linear_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "Linear models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_linear_model(
            reward_type=reward_type,
            penalty=model_params.get("penalty"),
            alpha=model_params.get("alpha"),
        )
        model_spec = None
    elif model_type == "gbdt_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "GBDT models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_gbdt(
            reward_type=reward_type,
            learning_rate=model_params["learning_rate"],
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
        )
        model_spec = None
    elif model_type == "random_forest_bandit":
        assert utils.pset_features_have_dense(feature_config["features"]), (
            "Random forest models require that product set features have associated"
            "dense representations."
        )
        model = model_constructors.build_random_forest(
            reward_type=reward_type,
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
        )
        model_spec = None

    # build the predictor
    predictor = BanditPredictor(
        feature_config=feature_config,
        float_feature_order=data["float_feature_order"],
        id_feature_order=data["id_feature_order"],
        id_feature_str_to_int_map=data["id_feature_str_to_int_map"],
        transforms=data["transforms"],
        imputers=data["imputers"],
        model=model,
        model_type=model_type,
        reward_type=reward_type,
        model_spec=model_spec,
        dense_features_to_use=dense_features_to_use,
    )

    # train the model
    if model_type == "neural_bandit":
        logger.info(f"Training {model_type} for {model_params['max_epochs']} epochs")
        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=reward_type,
            model=predictor.model,
            X=X,
            y=y,
            hyperparams=model_params,
            train_percent=ml_config["train_percent"],
        )
    elif model_type in ("gbdt_bandit", "random_forest_bandit", "linear_bandit"):
        logger.info(f"Training {model_type}")
        sklearn_model, _ = model_trainers.fit_sklearn_model(
            reward_type=reward_type,
            model=model,
            X=X,
            y=y,
            train_percent=ml_config["train_percent"],
        )

    if predictor_save_dir is not None:
        logger.info("Saving predictor artifacts to disk...")
        model_name = ml_config.get("model_name", "model")

        save_dir = f"{predictor_save_dir}/{model_name}/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        predictor_net_path = f"{save_dir}/{model_name}.pt"
        predictor_config_path = f"{save_dir}/{model_name}.json"
        predictor.config_to_file(predictor_config_path)
        predictor.model_to_file(predictor_net_path)
        shutil.make_archive(save_dir, "zip", save_dir)

    logger.info(f"Traning took {time.time() - start} seconds.")

    return predictor
