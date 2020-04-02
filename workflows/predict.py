"""
Example script showing how to use a saved model object
to make one off predictions.

Usage:
    python -m workflows.predict \
    	--predictor_dir trained_models/test-experiment-height-prediction-v8 \
        --model_name model_v1
"""

import argparse
import json
import pickle
import time
import zipfile

import pandas as pd

from banditml_pkg.banditml.serving.predictor import BanditPredictor
from utils.utils import get_logger


logger = get_logger(__name__)


def get_decisions(json_input, predictor):
    """Function that simulates a real time Python service making
    a prediction."""
    input = json.loads(json_input)
    return predictor.predict(input)


def main(args):
    config_path = f"{args.predictor_dir}/{args.model_name}.json"
    net_path = f"{args.predictor_dir}/{args.model_name}.pt"
    predictor = BanditPredictor.predictor_from_file(config_path, net_path)

    json_input = json.dumps({"year": 2019, "country": 4})

    start = time.time()
    decisions = get_decisions(json_input, predictor)
    end = time.time()

    logger.info(f"Prediction request took {round(time.time() - start, 5)} seconds.")
    logger.info(f"Predictions: {decisions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_dir", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    args = parser.parse_args()
    main(args)
