from collections import defaultdict
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.impute import SimpleImputer
from sklearn.utils import shuffle
import torch
from torch.nn.utils.rnn import pad_sequence


def get_preprocess_feature_order(features_spec: Dict) -> List[str]:
    """Get order that features should be fed into the preprocessor. This will
    need to match bandit-app. For consistency, we will enforce the following
    ordering logic:

    Split features into float_features (i.e. N, C )& id_features (i.e. P)
        - 1st order by feature type ["N", "C"] & ["P"]
        - 2nd order by alphabetical on feature_name a->z
    """
    N, C, P = [], [], []
    for feature_name, meta in features_spec.items():
        if meta["type"] == "N":
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
    feature_name: str, feature_type: str, values: np.array
) -> Tuple[pd.DataFrame, int]:
    assert feature_type in ("N", "C")

    # convert to scikit learn expected format
    values = values.reshape(-1, 1)

    if feature_type == "N":
        # standard scaler seems to be right choice for numeric features
        # in regression problems
        # http://rajeshmahajan.com/standard-scaler-v-min-max-scaler-machine-learning/
        imputer = SimpleImputer(strategy="mean")
        values = imputer.fit_transform(values)
        preprocessor = preprocessing.StandardScaler()
        values = preprocessor.fit_transform(values)
        df = pd.DataFrame(values.squeeze(), columns=[feature_name])
    elif feature_type == "C":
        imputer = SimpleImputer(strategy="most_frequent")
        values = imputer.fit_transform(values)
        preprocessor = preprocessing.OneHotEncoder(sparse=False)
        values = preprocessor.fit_transform(values)
        preprocessor.col_names = [
            feature_name + "_" + str(i) for i in preprocessor.categories_[0]
        ]
        df = pd.DataFrame(values.squeeze(), columns=preprocessor.col_names)
    else:
        raise Exception(f"Feature type {feature_type} not supported.")

    return df, preprocessor, imputer


def preprocess_data(
    raw_data: pd.DataFrame,
    reward_function: Dict,
    experiment_params: Dict,
    shuffle_data=True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    # start by randomizing the data upfront
    if shuffle_data:
        raw_data = shuffle(raw_data)

    # if context or decision is missing, this row is unsalvageble so drop it
    raw_data = raw_data.dropna(subset=["context", "decision"])

    # load the json string into json objects and expand into columns
    X = pd.json_normalize(raw_data["context"].apply(json.loads))
    X["decision"] = raw_data["decision"].values

    # fill in missing (NaN) rewards with empty metric maps
    raw_data["immediate_reward"] = raw_data["immediate_reward"].fillna("{}")
    immediate_y = pd.json_normalize(
        raw_data["immediate_reward"].apply(lambda x: json.loads(x))
    )
    immediate_y = immediate_y.fillna(0)

    # calculate the delayed rewards which requires joining them to their
    # previous decisions
    raw_data["end_of_mdp_reward"] = raw_data["delayed_reward"].fillna("{}")
    X["end_of_mdp_reward"] = (
        raw_data["end_of_mdp_reward"].apply(lambda x: json.loads(x)).values
    )
    delayed_rewards = X.apply(
        lambda row: row["end_of_mdp_reward"].get(str(row["decision"]), {}), axis=1
    )
    delayed_y = pd.json_normalize(delayed_rewards)
    delayed_y = delayed_y.fillna(0)

    # initialize the final reward scalar with 0s
    X["reward"] = np.zeros(len(X))

    # then add the linear combination of immediate rewards
    for metrics_name, series in immediate_y.iteritems():
        X["reward"] += series * reward_function[metrics_name]

    # then add the linear combination of delayed rewards
    for metrics_name, series in delayed_y.iteritems():
        X["reward"] += series * reward_function[metrics_name]

    reward_df = X["reward"]
    float_feature_df = pd.DataFrame()
    id_list_feature_df = pd.DataFrame()

    # order in which to preprocess features
    float_feature_order, id_feature_order = get_preprocess_feature_order(
        experiment_params["features"]
    )
    # final features names after preprocessing & expansion of categoricals
    final_float_feature_order, final_id_feature_order = [], []

    # create product set feature mappings for any product sets
    id_feature_str_to_int_map = {}
    for product_set, metadata in experiment_params["product_sets"].items():
        # index 0 in embedding tables is reserved for null id so + 1 below
        id_feature_str_to_int_map[product_set] = {
            v: idx + 1 for idx, v in enumerate(metadata["ids"])
        }

    transforms, imputers = {}, {}
    for feature_name in float_feature_order:
        meta = experiment_params["features"][feature_name]
        df, preprocessor, imputer = preprocess_feature(
            feature_name, meta["type"], X[feature_name].values
        )
        final_float_feature_order.extend(df.columns)
        float_feature_df = pd.concat([float_feature_df, df], axis=1)
        transforms[feature_name] = preprocessor
        imputers[feature_name] = imputer

    for feature_name in id_feature_order:
        meta = experiment_params["features"][feature_name]
        products_set_id = meta["product_set_id"]
        product_set_meta = experiment_params["product_sets"][products_set_id]

        if meta["use_dense"] is True and "dense" in product_set_meta:

            dense = defaultdict(list)
            # TODO: don't like that this is O(n^2), think about better way to do this
            for val in X[feature_name].values:
                for idx, feature_spec in enumerate(product_set_meta["features"]):
                    dense_feature_name = feature_spec["name"]
                    dense_feature_val = product_set_meta["dense"][val][idx]
                    dense[dense_feature_name].append(dense_feature_val)

            for idx, feature_spec in enumerate(product_set_meta["features"]):
                dtype = (
                    np.dtype(float) if feature_spec["type"] == "N" else np.dtype(object)
                )
                vals = np.array(dense[feature_spec["name"]], dtype=dtype)
                df, preprocessor, imputer = preprocess_feature(
                    feature_spec["name"], feature_spec["type"], vals
                )
                final_float_feature_order.extend(df.columns)
                float_feature_df = pd.concat([float_feature_df, df], axis=1)
                transforms[feature_spec["name"]] = preprocessor
                imputers[feature_spec["name"]] = imputer
        else:
            # sparse id list features need to be converted from string to int,
            # but aside from that are not imputed or transformed.
            product_set_id = experiment_params["features"][feature_name][
                "product_set_id"
            ]
            str_to_int_map = id_feature_str_to_int_map[product_set_id]
            final_id_feature_order.append(feature_name)
            id_list_feature_df[feature_name] = pd.Series(X[feature_name].values).apply(
                lambda x: str_to_int_map[x]
            )
            transforms[feature_name] = product_set_id
            imputers[feature_name] = product_set_id

    return {
        "y": reward_df,
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
