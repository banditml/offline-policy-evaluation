"""
Usage:
    python -m workflows.train_bandit \
        --ml_config_path configs/example_ml_config.json \
        --experiment_config_path configs/example_exp_config.json \
        --predictor_save_dir trained_models
"""

import argparse
import json
import os
import shutil
import sys
import time
from typing import Dict

from ..data_reader.reader import BigQueryReader
from ..preprocessing import preprocessor
from ..serving.predictor import BanditPredictor
from ..utils import (feature_importance, model_constructors, model_trainers,
                     utils)

logger = utils.get_logger(__name__)


def num_float_dim(data):
    return len(data["X_float"].columns)


def train(ml_params: Dict, experiment_params: Dict, predictor_save_dir: str = None):

    logger.info("Initializing data reader...")
    data_reader = BigQueryReader(
        credential_path=ml_params["data_reader"]["credential_path"],
        bq_project=ml_params["data_reader"]["bq_project"],
        bq_dataset=ml_params["data_reader"]["bq_dataset"],
        decisions_ds_start=ml_params["data_reader"]["decisions_ds_start"],
        decisions_ds_end=ml_params["data_reader"]["decisions_ds_end"],
        rewards_ds_end=ml_params["data_reader"]["rewards_ds_end"],
        reward_function=ml_params["data_reader"]["reward_function"],
        experiment_id=experiment_params["experiment_id"],
    )

    raw_data = data_reader.get_training_data()

    if len(raw_data) == 0:
        logger.error(f"Got no raws of training data. Training aborted.")
        sys.exit()
    logger.info(f"Got {len(raw_data)} rows of training data.")
    logger.info(raw_data.head())

    utils.fancy_print("Kicking off data preprocessing")

    # always add decision as a feature to use if not using all features
    features_to_use = ml_params["data_reader"].get("features_to_use", ["*"])
    if features_to_use != ["*"]:
        features_to_use.append(preprocessor.DECISION_FEATURE_NAME)
    features_to_use = list(set(features_to_use))
    dense_features_to_use = ml_params["data_reader"].get("dense_features_to_use", ["*"])

    data = preprocessor.preprocess_data(
        raw_data,
        experiment_params,
        ml_params["reward_type"],
        features_to_use,
        dense_features_to_use,
    )
    X, y = preprocessor.data_to_pytorch(data)

    model_type = ml_params["model_type"]
    model_params = ml_params["model_params"][model_type]
    reward_type = ml_params["reward_type"]

    feature_importance_params = ml_params.get("feature_importance", {})
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
            feature_specs=experiment_params["features"],
            product_sets=experiment_params["product_sets"],
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
        assert utils.pset_features_have_dense(experiment_params["features"]), (
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
        assert utils.pset_features_have_dense(experiment_params["features"]), (
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
        assert utils.pset_features_have_dense(experiment_params["features"]), (
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
        experiment_params=experiment_params,
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
            train_percent=ml_params["train_percent"],
        )
    elif model_type in ("gbdt_bandit", "random_forest_bandit", "linear_bandit"):
        logger.info(f"Training {model_type}")
        sklearn_model, _ = model_trainers.fit_sklearn_model(
            reward_type=reward_type,
            model=model,
            X=X,
            y=y,
            train_percent=ml_params["train_percent"],
        )

    if predictor_save_dir is not None:
        logger.info("Saving predictor artifacts to disk...")
        experiment_id = experiment_params.get("experiment_id", "test")
        model_name = ml_params.get("model_name", "model")

        save_dir = f"{predictor_save_dir}/{experiment_id}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        predictor_net_path = f"{save_dir}/{model_name}.pt"
        predictor_config_path = f"{save_dir}/{model_name}.json"
        predictor.config_to_file(predictor_config_path)
        predictor.model_to_file(predictor_net_path)
        shutil.make_archive(save_dir, "zip", save_dir)


def main(args):
    start = time.time()
    utils.fancy_print("Starting workflow", color="green", size=70)
    ml_params = utils.read_config(args.ml_config_path)
    utils.validate_ml_params(ml_params)

    if args.experiment_config_path:
        experiment_params = utils.read_config(args.experiment_config_path)
    else:
        assert args.experiment_id is not None, (
            "If no --experiment_config_path provided, --experiment_id must"
            " be provided to fetch experiment config from bandit app."
        )
        assert "bandit_app_credential_path" in ml_params, (
            "If getting experiment config from banditml.com, must provide"
            " valid api key path in `bandit_app_credential_path` in ml config."
        )
        logger.info("Getting experiment config from banditml.com...")
        experiment_params = utils.get_experiment_config_from_bandit_app(
            ml_params["bandit_app_credential_path"], args.experiment_id
        )
        experiment_params["experiment_id"] = args.experiment_id

    logger.info("Using parameters: {}\n".format(ml_params))
    train(
        ml_params=ml_params,
        experiment_params=experiment_params,
        predictor_save_dir=args.predictor_save_dir,
    )
    logger.info("Workflow completed successfully.")
    logger.info(f"Took {time.time() - start} seconds to complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml_config_path", required=True, type=str)
    parser.add_argument("--experiment_config_path", required=False, type=str)
    parser.add_argument("--experiment_id", required=False, type=str)
    parser.add_argument("--predictor_save_dir", required=False, type=str)
    args = parser.parse_args()
    main(args)
