"""
Example script showing how to use a saved model object
to make one off predictions.

Usage:
    python -m workflows.predict \
    	--predictor_dir trained_models/test-experiment-height-prediction-v1 \
        --model_name model_v1 \
        --get_ucb_scores
"""

import argparse
import json
import time

import numpy as np
from sklearn.preprocessing import scale

from banditml_pkg.banditml.serving.predictor import BanditPredictor
from utils.utils import get_logger

logger = get_logger(__name__)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_single_decision(decisions):
    scores = np.ndarray.flatten(np.array(decisions["scores"]))
    scores = softmax(scale(scores, with_mean=False, with_std=True))

    return {
        "greedyranker_decision": decisions["ids"][np.argmax(scores)],
        "softmaxranker_decision": decisions["ids"][
            np.random.choice(len(scores), p=scores)
        ],
    }


def get_decisions(json_input, predictor, get_ucb_scores=False):
    """Function that simulates a real time Python service making
    a prediction."""
    input = json.loads(json_input)
    return predictor.predict(input, get_ucb_scores)


def main(args):
    config_path = f"{args.predictor_dir}/{args.model_name}.json"
    net_path = f"{args.predictor_dir}/{args.model_name}.pt"
    predictor = BanditPredictor.predictor_from_file(config_path, net_path)

    json_input = json.dumps({"year": 2019, "country": "serbia"})

    start = time.time()
    decisions = get_decisions(json_input, predictor, args.get_ucb_scores)
    end = time.time()

    logger.info(f"Prediction request took {round(end - start, 5)} seconds.")
    logger.info(f"Predictions: {decisions}")
    if args.get_exploration_decision:
        logger.info(
            f"Single exploitation/exploration decision: {get_single_decision(decisions)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor_dir", required=True, type=str)
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--get_ucb_scores", action="store_true")
    parser.add_argument("--get_exploration_decision", action="store_true")
    args = parser.parse_args()
    main(args)
