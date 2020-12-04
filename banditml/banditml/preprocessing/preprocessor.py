import json
import logging
import statistics
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
from torch.nn.utils.rnn import pad_sequence

logging.basicConfig(
    format="[%(asctime)s %(levelname)-3s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MISSING_CATEGORICAL_CATEGORY = "null"

# special features with reserved names
DECISION_FEATURE_NAME = "decision"
POSITION_FEATURE_NAME = "position"

NUMPY_NUMERIC_TYPE = np.number
NUMPY_CATEGORICAL_TYPE = np.str_


def get_preprocess_feature_order(
    features_spec: Dict, features_to_use: List[str]
) -> List[str]:
    """Get order that features should be fed into the preprocessor. This will
    need to match bandit-app. For consistency, we will enforce the following
    ordering logic:

    Split features into float_features (i.e. N, C )& id_features (i.e. P)
        - 1st order by feature type ["N", "C"] & ["P"]
        - 2nd order by alphabetical on feature_name a->z
    """

    N, C, P = [], [], []
    for feature_name, meta in features_spec.items():
        if feature_name not in features_to_use and features_to_use != ["*"]:
            pass
        elif meta["type"] == "N":
            N.append(feature_name)
        elif meta["type"] == "C":
            C.append(feature_name)
        elif meta["type"] == "P":
            P.append(feature_name)
        else:
            raise Exception(f"Feature type {meta['type']} not supported.")

    N.sort()
    C.sort()
    P.sort()

    float_feature_order = N + C
    id_feature_order = P
    return float_feature_order, id_feature_order


def preprocess_feature(
    feature_name: str,
    feature_type: str,
    values: np.array,
    is_pset_dense_feature: bool = False,
) -> Tuple[pd.DataFrame, int]:
    assert feature_type in ("N", "C")

    # if this is a dense feature of a product set, make it clear in the logs
    prefix = "  --> " if is_pset_dense_feature else ""

    # convert to scikit learn expected format
    values = values.reshape(-1, 1)

    if feature_type == "N":
        # numeric features should be... numbers
        if not np.issubdtype(values.dtype, NUMPY_NUMERIC_TYPE):
            logger.warning(
                f"{feature_name} of type `N` has values of type {values.dtype}, "
                "but it should have numeric values. Attempting to cast it, but "
                "this might cause issues..."
            )
            values = values.astype(NUMPY_NUMERIC_TYPE)

        # standard scaler seems to be right choice for numeric features
        # in regression problems
        # http://rajeshmahajan.com/standard-scaler-v-min-max-scaler-machine-learning/
        imputer = SimpleImputer(strategy="mean")
        values = imputer.fit_transform(values)
        min_val, max_val = np.min(values), np.max(values)
        logger.info(f"{prefix}{feature_name} [{feature_type}]: [{min_val}, {max_val}]")
        preprocessor = preprocessing.StandardScaler()
        values = preprocessor.fit_transform(values)
        df = pd.DataFrame(values.squeeze(), columns=[feature_name])
    elif feature_type == "C":

        # categorical features should be strings
        if not values.dtype.type is NUMPY_CATEGORICAL_TYPE:
            logger.warning(
                f"{feature_name} of type `C` has values of type {values.dtype}, "
                "but it should have string values. Attempting to cast it, but "
                "this might cause issues..."
            )
            values = values.astype(NUMPY_CATEGORICAL_TYPE)

        imputer = None
        preprocessor = preprocessing.OneHotEncoder(sparse=False)
        # always add "null" as a possible value to categorical features
        # so a missing value is fine during inference even if a missing
        # value was never seen during traning (expect for the decision feature).
        if feature_name == DECISION_FEATURE_NAME:
            fit_values = values
        else:
            fit_values = np.append(values, [[MISSING_CATEGORICAL_CATEGORY]], axis=0)

        possible_values = set(fit_values.squeeze().tolist())
        logger.info(f"{prefix}{feature_name} [{feature_type}]: {possible_values}")

        preprocessor.fit(fit_values)
        values = preprocessor.transform(values)
        preprocessor.col_names = [
            feature_name + "_" + i for i in preprocessor.categories_[0]
        ]
        df = pd.DataFrame(values.squeeze(), columns=preprocessor.col_names)
    else:
        raise Exception(f"Feature type {feature_type} not supported.")

    return df, preprocessor, imputer


def preprocess_data(
    raw_data: pd.DataFrame,
    feature_config: Dict,
    reward_type: str,
    features_to_use: List[str] = ["*"],
    dense_features_to_use: List[str] = ["*"],
    shuffle_data: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # start by randomizing the data upfront
    if shuffle_data:
        raw_data = shuffle(raw_data)

    # if context or decision is missing, this row is unsalvageble so drop it
    raw_data = raw_data.dropna(subset=["context", DECISION_FEATURE_NAME])

    # load the json string into json objects and expand into columns
    X = pd.json_normalize(raw_data["context"].apply(json.loads))
    X[DECISION_FEATURE_NAME] = raw_data[DECISION_FEATURE_NAME].values
    X["reward"] = raw_data["reward"].astype(float).values

    float_feature_df = pd.DataFrame()
    id_list_feature_df = pd.DataFrame()

    # output how sparse the reward is to help user during training
    total_rows = len(X)
    non_zero_reward_rows = sum(X["reward"] != 0)
    percent_non_zero = round(non_zero_reward_rows / total_rows * 100, 2)
    logger.info("Reward stats")
    logger.info(_sub_dividing_text())
    logger.info(
        f"{non_zero_reward_rows} of {total_rows} rows have non-zero reward "
        f"({percent_non_zero}%)"
    )

    max_reward = max(X["reward"])
    min_reward = min(X["reward"])
    logger.info(f"Reward range: [{min_reward}, {max_reward}]")
    if reward_type == "binary" and max_reward > 1:
        logger.warning("Reward type specified as binary, but reward > 1 found.")
        logger.warning("Coverting rewards to binary values.")
        X["reward"] = (X["reward"] > 0) * 1

    logger.info("Manually selected features")
    logger.info(_sub_dividing_text())
    # order in which to preprocess features
    float_feature_order, id_feature_order = get_preprocess_feature_order(
        feature_config["features"], features_to_use
    )
    # final features names after preprocessing & expansion of categoricals
    final_float_feature_order, final_id_feature_order = [], []

    # create product set feature mappings for any product sets
    id_feature_str_to_int_map = {}
    for product_set, metadata in feature_config["product_sets"].items():
        # index 0 in embedding tables is reserved for null id so + 1 below
        id_feature_str_to_int_map[product_set] = {
            v: idx + 1 for idx, v in enumerate(metadata["ids"])
        }

    transforms, imputers = {}, {}
    for feature_name in float_feature_order:
        meta = feature_config["features"][feature_name]

        # rather than using a `most_frequent` imputation for categorical features, I
        # think just adding another category of "null" is actually a better idea.
        if meta["type"] == "C":
            X[feature_name].fillna(value=MISSING_CATEGORICAL_CATEGORY, inplace=True)

        df, preprocessor, imputer = preprocess_feature(
            feature_name, meta["type"], X[feature_name].values
        )
        final_float_feature_order.extend(df.columns)
        float_feature_df = pd.concat([float_feature_df, df], axis=1)
        transforms[feature_name] = preprocessor
        imputers[feature_name] = imputer

    for feature_name in id_feature_order:
        meta = feature_config["features"][feature_name]
        products_set_id = meta["product_set_id"]
        product_set_meta = feature_config["product_sets"][products_set_id]
        logger.info(f"{feature_name} [{meta['type']}]")

        if meta["use_dense"] is True and "dense" in product_set_meta:

            dense = defaultdict(list)
            # TODO: don't like that this is O(n^2), think about better way to do this
            for val in X[feature_name].values:
                if not isinstance(val, list):
                    val = [val]

                dense_matrix = []
                for v in val:
                    dense_features = product_set_meta["dense"].get(v)
                    if not dense_features:
                        logger.warning(
                            f"No dense representation found for '{feature_name}'"
                            f" product set value '{v}'."
                        )
                    else:
                        dense_matrix.append(dense_features)

                if not dense_matrix:
                    # there were no or no valid id features to add, add an
                    # empty row to be imputed
                    dense_matrix.append([])

                for idx, feature_spec in enumerate(product_set_meta["features"]):
                    dense_feature_name = feature_spec["name"]
                    row_vals = []
                    for row in dense_matrix:
                        if not row:
                            dense_feature_val = (
                                MISSING_CATEGORICAL_CATEGORY
                                if feature_spec["type"] == "C"
                                else None
                            )
                        else:
                            dense_feature_val = row[idx]
                        row_vals.append(dense_feature_val)

                    dense_feature_val = flatten_dense_id_list_feature(
                        row_vals, feature_spec["type"]
                    )
                    dense[dense_feature_name].append(dense_feature_val)

            for idx, feature_spec in enumerate(product_set_meta["features"]):
                # prepend the id list feature name to prevent name collisions
                # with two features that map to same product set
                dense_feature_name_desc = f"{feature_name}:{feature_spec['name']}"

                if (
                    dense_features_to_use != ["*"]
                    and dense_feature_name_desc not in dense_features_to_use
                ):
                    continue

                dtype = (
                    NUMPY_NUMERIC_TYPE
                    if feature_spec["type"] == "N"
                    else NUMPY_CATEGORICAL_TYPE
                )

                vals = dense[feature_spec["name"]]
                if feature_spec["type"] == "C":
                    # fill in null categorical values with a "null" category
                    vals = [
                        MISSING_CATEGORICAL_CATEGORY if v is None else v for v in vals
                    ]

                vals = np.array(vals, dtype=dtype)
                df, preprocessor, imputer = preprocess_feature(
                    dense_feature_name_desc,
                    feature_spec["type"],
                    vals,
                    is_pset_dense_feature=True,
                )
                final_float_feature_order.extend(df.columns)
                float_feature_df = pd.concat([float_feature_df, df], axis=1)
                transforms[dense_feature_name_desc] = preprocessor
                imputers[dense_feature_name_desc] = imputer
        else:
            # sparse id list features need to be converted from string to int,
            # but aside from that are not imputed or transformed.
            product_set_id = feature_config["features"][feature_name]["product_set_id"]

            str_to_int_map = id_feature_str_to_int_map[product_set_id]

            def _str_id_list_converter(input):
                if not isinstance(input, list):
                    input = [input]
                out = []
                for i in input:
                    int_id = str_to_int_map.get(i)
                    if not int_id:
                        logger.info(f"{i} missing from product_set {product_set_id}")
                    else:
                        out.append(int_id)
                return out

            final_id_feature_order.append(feature_name)
            id_list_feature_df[feature_name] = pd.Series(X[feature_name].values).apply(
                lambda x: _str_id_list_converter(x)
            )
            transforms[feature_name] = None
            imputers[feature_name] = None

    return {
        "y": X["reward"],
        "X_float": float_feature_df,
        "X_id_list": id_list_feature_df,
        "transforms": transforms,
        "imputers": imputers,
        "id_feature_str_to_int_map": id_feature_str_to_int_map,
        "float_feature_order": float_feature_order,
        "id_feature_order": id_feature_order,
        "final_float_feature_order": final_float_feature_order,
        "final_id_feature_order": final_id_feature_order,
    }


def flatten_dense_id_list_feature(vals, feature_type):
    if feature_type == "C":
        # flatten categorical dense features by taking the mode
        return max(set(vals), key=vals.count)
    elif feature_type == "N":
        # drop nones and take mean across values remaining
        vals = [v for v in vals if v is not None]

        if not vals:
            return None

        return statistics.mean(vals)

    raise Exception(f"Feature type {feature_type} not supported.")


def data_to_pytorch(data: Dict):
    X = {}

    if len(data["X_float"]) > 0:
        X["X_float"] = torch.tensor(data["X_float"].values, dtype=torch.float32)

    y, X_id_list, X_id_list_idxs = None, None, None

    # hack needed due to skorch not handling objects in .fit() besides
    # a dict of lists or tensors (dict of dicts not supported.)
    id_list_pad_idxs = []
    pad_idx = 0

    for _, series in data["X_id_list"].iteritems():
        # wrap decisions in lists
        if series.dtype in (int, float):
            series = series.apply(lambda x: [x])

        pre_pad = [torch.tensor(i, dtype=torch.long) for i in series]
        post_pad = pad_sequence(pre_pad, batch_first=True, padding_value=0)
        idx_offset = post_pad.shape[1]
        id_list_pad_idxs.extend([pad_idx, pad_idx + idx_offset])
        pad_idx += idx_offset

        if X_id_list is not None:
            X_id_list = torch.cat((X_id_list, post_pad), dim=1)
        else:
            X_id_list = post_pad

    # skorch requires all inputs to .fit() to have the same length so we
    # need to repeat our id_list_pad_idxs list.
    if X_id_list is not None:
        X["X_id_list"] = X_id_list
        X["X_id_list_idxs"] = torch.tensor([id_list_pad_idxs for i in X_id_list])

    # this function is used to make predictions as well, where there is no y
    if "y" in data:
        y = torch.tensor(data["y"].values, dtype=torch.float32).unsqueeze(dim=1)

    return X, y


def _sub_dividing_text(size=40):
    return "-" * size
