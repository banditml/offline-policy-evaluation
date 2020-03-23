"""
Example script showing how to use a saved model object
to make one off predictions.

Usage:
    python -m workflows.predict \
    	--model_path trained_models/test_model.pkl
"""

import argparse
import json
import pickle
import time

import pandas as pd

from banditml_pkg.banditml.model_io import read_predictor_from_disk
from utils.utils import get_logger


logger = get_logger(__name__)


def get_decisions(json_input, model):
    """Function that simulates a real time Python service making
    a prediction."""
    input = json.loads(json_input)
    return model.predict(input)


def main(args):
    model = read_predictor_from_disk(args.model_path)
    json_input = json.dumps({"year": 2019, "country": 4})

    start = time.time()
    decisions = get_decisions(json_input, model)
    end = time.time()

    logger.info(f"Prediction request took {round(time.time() - start, 5)} seconds.")
    logger.info(f"Predictions: {decisions}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, type=str)
    args = parser.parse_args()
    main(args)
