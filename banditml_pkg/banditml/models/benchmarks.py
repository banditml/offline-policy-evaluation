"""
Out of the box sklearn models used to benchmark
the performance of custom PyTorch models.
"""

from sklearn import ensemble
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.neural_network import MLPRegressor


def fit_sklearn_mlp_regression(X_train, y_train, X_test, y_test, hyperparams):
    """Fit an off the shelf sklearn MLP regressor. Used to validate that our
    models are compeitive."""

    clf = MLPRegressor(
        hidden_layer_sizes=(hyperparams["layers"][1], hyperparams["layers"][2]),
        activation=hyperparams["activations"][0],
        solver="adam",
        alpha=hyperparams["l2_decay"],
        batch_size=hyperparams["batch_size"],
        learning_rate_init=hyperparams["learning_rate"],
        tol=1e-2,
        max_iter=hyperparams["max_epochs"],
        shuffle=True,
        verbose=False,
        early_stopping=False,
        validation_fraction=0,
    )
    clf.fit(X_train, y_train)
    mse_train = mean_squared_error(y_train, clf.predict(X_train))
    mse_test = mean_squared_error(y_test, clf.predict(X_test))
    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_sklearn_gbdt_regression(X_train, y_train, X_test, y_test, hyperparams):
    """Fit an off the shelf sklearn GBDT regressor. Used to validate that our
    models are compeitive."""

    params = {
        "n_estimators": 500,
        "max_depth": 4,
        "min_samples_split": 2,
        "learning_rate": 0.01,
        "loss": "ls",
    }

    clf = ensemble.GradientBoostingRegressor(**params)
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))
    mse_test = mean_squared_error(y_test, clf.predict(X_test))
    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_sklearn_gbdt_classification(X_train, y_train, X_test, y_test, hyperparams):
    """Fit an off the shelf sklearn GBDT classifier. Used to validate that our
    models are compeitive."""

    clf = ensemble.GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))
    acc_test = accuracy_score(y_test, clf.predict(X_test))
    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}
