import os
from copy import deepcopy

import pandas as pd

from banditml.banditml.preprocessing import preprocessor


class Params:

    # experiment params for country as ID list variable
    FEATURE_CONFIG_COUNTRY_AS_ID_LIST = {
        "choices": ["male", "female"],
        "features": {
            "country": {"type": "P", "product_set_id": "1", "use_dense": False},
            "year": {"type": "N"},
            "decision": {"type": "C"},
        },
        "product_sets": {
            "1": {
                "ids": ["usa", "china", "india", "serbia", "norway"],
                "dense": {
                    "usa": ["north-america", 10.0],
                    "china": ["asia", 8.5],
                    "india": ["asia", 7.5],
                    "serbia": ["europe", 11.5],
                    "norway": ["europe", 10.5],
                },
                "features": [
                    {"name": "region", "type": "C"},
                    {"name": "avg_shoe_size_m", "type": "N"},
                ],
            }
        },
    }

    # experiment params for country as categorical variable
    _tmp = deepcopy(FEATURE_CONFIG_COUNTRY_AS_ID_LIST)
    _tmp["features"]["country"] = {"type": "C"}
    FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL = _tmp

    # experiment params for country as dense ID list variable
    _tmp_1 = deepcopy(FEATURE_CONFIG_COUNTRY_AS_ID_LIST)
    _tmp_1["features"]["country"] = {
        "type": "P",
        "product_set_id": "1",
        "use_dense": True,
    }
    FEATURE_CONFIG_COUNTRY_AS_DENSE_ID_LIST = _tmp_1

    # experiment params for country as ID list AND decision as ID list variables
    _tmp_2 = deepcopy(FEATURE_CONFIG_COUNTRY_AS_ID_LIST)
    _tmp_2["features"]["decision"] = {
        "type": "P",
        "product_set_id": "2",
        "use_dense": False,
    }
    _tmp_2["product_sets"]["2"] = {
        "ids": ["male", "female"],
        "dense": {"male": ["male"], "female": ["female"]},
        "features": [{"name": "gender", "type": "C"}],
    }
    FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST = _tmp_2

    ML_CONFIG = {
        "data_reader": {"reward_function": {"height": 1, "nameOfADelayedReward": 1}},
        "reward_type": "regression",
        "model_type": "neural_bandit",
        "model_params": {
            "neural_bandit": {
                "max_epochs": 50,
                "learning_rate": 0.002,
                "l2_decay": 0.0003,
                "batch_size": 64,
                "layers": [-1, 32, 16, -1],
                "activations": ["relu", "relu", "linear"],
                "dropout_ratio": 0.00,
            }
        },
        "train_percent": 0.8,
    }

    REWARD_FUNCTION_BINARY = {"taller_than_165": 1, "nameOfADelayedReward": 0}


class Datasets:
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DATASET_DIR = "datasets"

    # continuous reward
    TEST_DATASET_FILENAME = "height_dataset.csv"
    DATASET_PATH = os.path.join(TEST_DIR, TEST_DATASET_DIR, TEST_DATASET_FILENAME)

    _raw_data = pd.read_csv(DATASET_PATH)
    _offset = int(len(_raw_data) * Params.ML_CONFIG["train_percent"])

    # dataset for country as categorical variable
    DATA_COUNTRY_CATEG = preprocessor.preprocess_data(
        _raw_data,
        Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL,
        Params.ML_CONFIG["reward_type"],
    )
    _X, _y = preprocessor.data_to_pytorch(DATA_COUNTRY_CATEG)
    X_COUNTRY_CATEG = {
        "X_train": {"X_float": _X["X_float"][:_offset]},
        "y_train": _y[:_offset],
        "X_test": {"X_float": _X["X_float"][_offset:]},
        "y_test": _y[_offset:],
    }

    # dataset for country as ID list variable
    DATA_COUNTRY_ID_LIST = preprocessor.preprocess_data(
        _raw_data,
        Params.FEATURE_CONFIG_COUNTRY_AS_ID_LIST,
        Params.ML_CONFIG["reward_type"],
    )
    _X, _y = preprocessor.data_to_pytorch(DATA_COUNTRY_ID_LIST)
    X_COUNTRY_ID_LIST = {
        "X_train": {
            "X_float": _X["X_float"][:_offset],
            "X_id_list": _X["X_id_list"][:_offset],
            "X_id_list_idxs": _X["X_id_list_idxs"][:_offset],
        },
        "y_train": _y[:_offset],
        "X_test": {
            "X_float": _X["X_float"][_offset:],
            "X_id_list": _X["X_id_list"][_offset:],
            "X_id_list_idxs": _X["X_id_list_idxs"][_offset:],
        },
        "y_test": _y[_offset:],
    }

    # dataset for country as dense ID list variable
    DATA_COUNTRY_DENSE_ID_LIST = preprocessor.preprocess_data(
        _raw_data,
        Params.FEATURE_CONFIG_COUNTRY_AS_DENSE_ID_LIST,
        Params.ML_CONFIG["reward_type"],
    )
    _X, _y = preprocessor.data_to_pytorch(DATA_COUNTRY_DENSE_ID_LIST)
    X_COUNTRY_DENSE_ID_LIST = {
        "X_train": {"X_float": _X["X_float"][:_offset]},
        "y_train": _y[:_offset],
        "X_test": {"X_float": _X["X_float"][_offset:]},
        "y_test": _y[_offset:],
    }

    # dataset for country as ID list AND decision as ID list variables
    DATA_COUNTRY_AND_DECISION_ID_LIST = preprocessor.preprocess_data(
        _raw_data,
        Params.FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST,
        Params.ML_CONFIG["reward_type"],
    )
    _X, _y = preprocessor.data_to_pytorch(DATA_COUNTRY_AND_DECISION_ID_LIST)
    X_COUNTRY_AND_DECISION_ID_LIST = {
        "X_train": {
            "X_float": _X["X_float"][:_offset],
            "X_id_list": _X["X_id_list"][:_offset],
            "X_id_list_idxs": _X["X_id_list_idxs"][:_offset],
        },
        "y_train": _y[:_offset],
        "X_test": {
            "X_float": _X["X_float"][_offset:],
            "X_id_list": _X["X_id_list"][_offset:],
            "X_id_list_idxs": _X["X_id_list_idxs"][_offset:],
        },
        "y_test": _y[_offset:],
    }

    # binary reward
    TEST_BINARY_REWARD_DATASET_FILENAME = "height_dataset_binary.csv"
    BINARY_REWARD_DATASET_PATH = os.path.join(
        TEST_DIR, TEST_DATASET_DIR, TEST_BINARY_REWARD_DATASET_FILENAME
    )

    _raw_data_binary_reward = pd.read_csv(BINARY_REWARD_DATASET_PATH)
    _offset_binary_reward = int(
        len(_raw_data_binary_reward) * Params.ML_CONFIG["train_percent"]
    )

    # dataset for country as categorical variable & binary reward
    DATA_COUNTRY_CATEG_BINARY_REWARD = preprocessor.preprocess_data(
        _raw_data_binary_reward,
        Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL,
        "binary",
    )
    _X_binary_reward, _y_binary_reward = preprocessor.data_to_pytorch(
        DATA_COUNTRY_CATEG_BINARY_REWARD
    )
    X_COUNTRY_CATEG_BINARY_REWARD = {
        "X_train": {"X_float": _X_binary_reward["X_float"][:_offset_binary_reward]},
        "y_train": _y_binary_reward[:_offset_binary_reward],
        "X_test": {"X_float": _X_binary_reward["X_float"][_offset_binary_reward:]},
        "y_test": _y_binary_reward[_offset_binary_reward:],
    }
