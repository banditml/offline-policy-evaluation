from copy import deepcopy
import os

import pandas as pd

from banditml_pkg.banditml.preprocessing import preprocessor


class Params:

    # experiment params for country as ID list variable
    EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST = {
        "features": {
            "country": {
                "type": "P",
                "possible_values": None,
                "product_set_id": "1",
                "use_dense": False,
            },
            "year": {"type": "N", "possible_values": None, "product_set_id": None},
            "decision": {
                "type": "C",
                "possible_values": [1, 2],
                "product_set_id": None,
            },
        },
        "reward_function": {"height": 1, "nameOfADelayedReward": 1},
        "product_sets": {
            "1": {
                "ids": [1, 2, 3, 4, 5],
                "dense": {
                    "1": [0, 10.0],
                    "2": [1, 8.5],
                    "3": [1, 7.5],
                    "4": [2, 11.5],
                    "5": [2, 10.5],
                },
                "features": [
                    {"name": "region", "type": "C", "possible_values": [0, 1, 2]},
                    {"name": "avg_shoe_size_m", "type": "N", "possible_values": None},
                ],
            }
        },
    }

    # experiment params for country as categorical variable
    _tmp = deepcopy(EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST)
    _tmp["features"]["country"] = {
        "type": "C",
        "possible_values": [1, 2, 3, 4, 5],
        "product_set_id": None,
    }
    EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL = _tmp

    # experiment params for country as dense ID list variable
    _tmp_1 = deepcopy(EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST)
    _tmp_1["features"]["country"] = {
        "type": "P",
        "possible_values": None,
        "product_set_id": "1",
        "use_dense": True,
    }
    EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST = _tmp_1

    # experiment params for country as ID list AND decision as ID list variables
    _tmp_2 = deepcopy(EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST)
    _tmp_2["features"]["decision"] = {
        "type": "P",
        "possible_values": [1, 2],
        "product_set_id": "2",
        "use_dense": False,
    }
    _tmp_2["product_sets"]["2"] = {
        "ids": [1, 2],
        "dense": {1: [1], 2: [2]},
        "features": [{"name": "gender", "type": "C", "possible_values": [1, 2]}],
    }
    EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST = _tmp_2

    # shared model params
    SHARED_PARAMS = {
        "data_reader": {},
        "max_epochs": 50,
        "learning_rate": 0.002,
        "l2_decay": 0.0003,
        "batch_size": 64,
        "model": {
            "layers": [-1, 32, 16, -1],
            "activations": ["relu", "relu", "linear"],
        },
        "train_test_split": 0.8,
    }


class Datasets:
    TEST_DIR = os.path.dirname(os.path.abspath(__file__))
    TEST_DATASET_DIR = "datasets"
    TEST_DATASET_FILENAME = "height_dataset.csv"
    DATASET_PATH = os.path.join(TEST_DIR, TEST_DATASET_DIR, TEST_DATASET_FILENAME)

    _raw_data = pd.read_csv(DATASET_PATH)
    _offset = int(len(_raw_data) * Params.SHARED_PARAMS["train_test_split"])

    # dataset for country as categorical variable
    DATA_COUNTRY_CATEG = preprocessor.preprocess_data(
        _raw_data, Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL
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
        _raw_data, Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST
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
        _raw_data, Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST
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
        _raw_data, Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST
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
