from typing import Dict

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score, mean_squared_error


def fit_ridge_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge regression."""

    clf = Ridge()
    clf.fit(X_train, y_train)

    mse_train = mean_squared_error(y_train, clf.predict(X_train))

    mse_test = None
    if X_test and y_test:
        mse_test = mean_squared_error(y_test, clf.predict(X_test))

    return {"model": clf, "mse_train": mse_train, "mse_test": mse_test}


def fit_ridge_classification(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray = None,
    y_test: np.ndarray = None,
) -> Dict:
    """Off the shelf sklearn Ridge classifier."""

    clf = RidgeClassifier()
    clf.fit(X_train, y_train)

    acc_train = accuracy_score(y_train, clf.predict(X_train))

    acc_test = None
    if X_test and y_test:
        acc_test = accuracy_score(y_test, clf.predict(X_test))

    return {"model": clf, "acc_train": acc_train, "acc_test": acc_test}
