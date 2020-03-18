"""
Usage:
    python -m workflows.train_bandit \
        --params_path configs/bandit.json \
        --experiment_config_path configs/example_experiment_config.json \
        --model_path trained_models/model.pkl
"""

import argparse
import time
from typing import Dict, List, NoReturn, Tuple

from skorch import NeuralNetRegressor
import torch

from data_reader.bandit_reader import BigQueryReader
from ml.preprocessing import preprocessor
from ml.models.embed_dnn import EmbedDnn
from ml.serving.predictor import BanditPredictor
from ml.serving.model_io import write_predictor_to_disk
from utils.utils import get_logger, fancy_print, read_config

logger = get_logger(__name__)


def build_pytorch_net(
    feature_specs,
    product_sets,
    float_feature_order,
    layers,
    id_feature_order,
    activations,
    input_dim,
    output_dim=1,
):
    """Build PyTorch model that will be fed into skorch training."""
    layers[0], layers[-1] = input_dim, output_dim
    return EmbedDnn(
        layers,
        activations,
        feature_specs=feature_specs,
        product_sets=product_sets,
        float_feature_order=float_feature_order,
        id_feature_order=id_feature_order,
    )


def num_float_dim(data):
    return len(data["X_float"].columns)


def fit_custom_pytorch_module_w_skorch(module, X, y, hyperparams):
    """Fit a custom PyTorch module using Skorch."""

    skorch_net = NeuralNetRegressor(
        module=module,
        optimizer=torch.optim.Adam,
        lr=hyperparams["learning_rate"],
        optimizer__weight_decay=hyperparams["l2_decay"],
        max_epochs=hyperparams["max_epochs"],
        batch_size=hyperparams["batch_size"],
        iterator_train__shuffle=True,
    )

    skorch_net.fit(X, y)
    return skorch_net


def train(
    shared_params: Dict, experiment_specific_params: Dict, model_path: str = None
):

    logger.info("Initializing data reader...")
    data_reader = BigQueryReader(
        credential_path=shared_params["data_reader"]["credential_path"],
        decisions_table_name=shared_params["data_reader"]["decisions_table_name"],
        rewards_table_name=shared_params["data_reader"]["rewards_table_name"],
        decisions_ds_start=experiment_specific_params["decisions_ds_start"],
        decisions_ds_end=experiment_specific_params["decisions_ds_end"],
        rewards_ds_end=experiment_specific_params["rewards_ds_end"],
        experiment_id=experiment_specific_params["experiment_id"],
    )

    raw_data = data_reader.get_training_data()
    logger.info(f"Got {len(raw_data)} rows of training data.")
    logger.info(raw_data.head())

    data = preprocessor.preprocess_data(raw_data, experiment_specific_params)
    X, y = preprocessor.data_to_pytorch(data)

    pytorch_net = build_pytorch_net(
        feature_specs=experiment_specific_params["features"],
        product_sets=experiment_specific_params["product_sets"],
        float_feature_order=data["final_float_feature_order"],
        id_feature_order=data["final_id_feature_order"],
        layers=shared_params["model"]["layers"],
        activations=shared_params["model"]["activations"],
        input_dim=num_float_dim(data),
    )
    logger.info(f"Initialized model: {pytorch_net}")

    logger.info(f"Starting training: {shared_params['max_epochs']} epochs")
    skorch_net = fit_custom_pytorch_module_w_skorch(
        module=pytorch_net, X=X, y=y, hyperparams=shared_params
    )

    if model_path is not None:
        predictor = BanditPredictor(
            experiment_specific_params=experiment_specific_params,
            float_feature_order=data["float_feature_order"],
            id_feature_order=data["id_feature_order"],
            transforms=data["transforms"],
            net=skorch_net,
        )

        write_predictor_to_disk(predictor, path=model_path)


def main(args):
    start = time.time()
    fancy_print("Starting workflow", color="green", size=70)
    wf_params = read_config(args.params_path)

    if args.experiment_config_path:
        experiment_specific_params = read_config(args.experiment_config_path)
    else:
        logger.info("Getting experiment config from banditml.com...")
        # TODO: read from banditml.com
        pass

    logger.info("Using parameters: {}".format(wf_params))
    train(wf_params, experiment_specific_params, args.model_path)
    logger.info("Workflow completed successfully.")
    logger.info(f"Took {time.time() - start} seconds to complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--params_path", required=True, type=str)
    parser.add_argument("--experiment_config_path", required=False, type=str)
    parser.add_argument("--model_path", required=False, type=str)
    args = parser.parse_args()
    main(args)
