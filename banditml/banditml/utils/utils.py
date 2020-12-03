import json
import logging
import math
from typing import Dict, NoReturn

import pandas as pd

VALID_MODEL_TYPES = (
    "linear_bandit",
    "neural_bandit",
    "random_forest_bandit",
    "gbdt_bandit",
    "dqn",
    "qr_dqn",
)
VALID_REWARD_TYPES = ("regression", "binary")


def read_config(config_path):
    with open(config_path) as json_file:
        config = json.load(json_file)
    return config


def get_logger(name: str):
    logging.basicConfig(
        format="[%(asctime)s %(levelname)-3s] %(message)s",
        level=logging.INFO,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(name)


def fancy_print(text: str, color="blue", size=60):
    ansi_color = "\033[94m"
    if color == "green":
        ansi_color = "\033[95m"
    elif color == "blue":
        ansi_color = "\033[94m"
    else:
        raise Exception(f"Color {color} not supported")

    end_color = "\033[0m"
    str_len = len(text)
    padding = math.ceil((size - str_len) / 2)
    header_len = padding * 2 + str_len + 2
    border = "#" * header_len
    message = "#" * padding + " " + text + " " + "#" * padding
    print(f"{ansi_color}\n{border}\n{message}\n{border}\n{end_color}")


def color_text(text: str, color="blue"):
    ansi_color = "\033[94m"
    if color == "green":
        ansi_color = "\033[95m"
    elif color == "blue":
        ansi_color = "\033[94m"
    else:
        raise Exception(f"Color {color} not supported")

    end_color = "\033[0m"
    return f"{ansi_color}{text}{end_color}"


def validate_ml_config(ml_config: Dict) -> NoReturn:
    assert "model_type" in ml_config
    assert "model_params" in ml_config
    assert "reward_type" in ml_config

    model_type = ml_config["model_type"]
    model_params = ml_config["model_params"]
    reward_type = ml_config["reward_type"]

    assert (
        model_type in VALID_MODEL_TYPES
    ), f"Model type {model_type} not supported. Valid model types are {VALID_MODEL_TYPES}"
    assert model_type in model_params
    assert (
        reward_type in VALID_REWARD_TYPES
    ), f"Reward type {reward_type} not supported. Valid reward types are {VALID_REWARD_TYPES}"


def validate_training_data_schema(training_df: pd.DataFrame) -> NoReturn:
    for col in ["context", "decision", "reward", "mdp_id", "sequence_number"]:
        assert col in training_df.columns


def pset_features_have_dense(features: Dict) -> NoReturn:
    for feature, meta in features.items():
        if meta["type"] == "P":
            if not meta["use_dense"]:
                return False
    return True
