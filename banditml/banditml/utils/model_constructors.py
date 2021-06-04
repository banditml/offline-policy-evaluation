from sklearn import ensemble, linear_model

from ..models.embed_dnn import EmbedDnn, build_embedding_spec


def build_pytorch_net(
    feature_specs,
    product_sets,
    float_feature_order,
    id_feature_order,
    reward_type,
    layers,
    activations,
    input_dim,
    dropout_ratio=0.0,
    is_mdn=False,
):
    """Build PyTorch model that will be fed into skorch training."""

    is_classification = reward_type == "binary"
    output_dim = 2 if is_classification else 1

    layers[0], layers[-1] = input_dim, output_dim

    # handle changes of model architecture due to embeddings
    first_layer_dim_increase, embedding_info = build_embedding_spec(
        id_feature_order, feature_specs, product_sets
    )
    layers[0] += first_layer_dim_increase

    model_spec = {
        "layers": layers,
        "activations": activations,
        "dropout_ratio": dropout_ratio,
        "feature_specs": feature_specs,
        "product_sets": product_sets,
        "float_feature_order": float_feature_order,
        "id_feature_order": id_feature_order,
        "embedding_info": embedding_info,
        "is_classification": is_classification,
        "is_mdn": is_mdn,
    }
    return model_spec, EmbedDnn(**model_spec)


def build_gbdt(reward_type, learning_rate=0.1, n_estimators=100, max_depth=3):
    is_classification = reward_type == "binary"
    params = {
        "learning_rate": learning_rate,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
    }

    if is_classification:
        gbdt_model = ensemble.GradientBoostingClassifier(**params)
    else:
        gbdt_model = ensemble.GradientBoostingRegressor(**params)

    return gbdt_model


def build_random_forest(reward_type, n_estimators=100, max_depth=None):
    is_classification = reward_type == "binary"
    params = {"n_estimators": n_estimators, "max_depth": max_depth}

    if is_classification:
        gbdt_model = ensemble.RandomForestClassifier(**params)
    else:
        gbdt_model = ensemble.RandomForestRegressor(**params)

    return gbdt_model


def build_linear_model(reward_type, penalty="l2", alpha=1.0):
    is_classification = reward_type == "binary"
    if is_classification:
        params = {"penalty": penalty}
        linear_model_ = linear_model.LogisticRegression(**params)
    else:
        params = {"alpha": alpha}
        linear_model_ = linear_model.Ridge(**params)

    return linear_model_
