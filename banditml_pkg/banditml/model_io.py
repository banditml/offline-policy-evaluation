"""
Module with functions to write/read models to/from different
locations & data stores.
"""

import dill as pickle
from typing import Dict, NoReturn


def write_predictor_to_disk(predictor, path) -> NoReturn:
    with open(path, "wb") as f:
        pickle.dump(predictor, f)


def read_predictor_from_disk(path) -> Dict:
    with open(path, "rb") as f:
        predictor = pickle.load(f)
    return predictor
