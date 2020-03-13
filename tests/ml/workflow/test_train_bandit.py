from copy import deepcopy
import os
import unittest

import pandas as pd

from ml.models import benchmarks
from workflows import train_bandit


TEST_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATASET_DIR = "datasets"
TEST_DATASET_FILENAME = "height_dataset.csv"
DATASET_PATH = os.path.join(TEST_DIR, TEST_DATASET_DIR, TEST_DATASET_FILENAME)


class TestTrainBandit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # country as ID list experiment parameters
        cls.experiment_specific_params_country_as_id_list = {
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
                    "possible_values": [0, 1],
                    "product_set_id": None,
                },
            },
            "reward_function": {"height": 1},
            "product_sets": {
                "1": {
                    "ids": [1, 2, 3, 4, 5],
                    "dense": {
                        1: [0, 10.0],
                        2: [1, 8.5],
                        3: [1, 7.5],
                        4: [2, 11.5],
                        5: [2, 10.5],
                    },
                    "features": [
                        {"name": "region", "type": "C", "possible_values": [0, 1, 2]},
                        {
                            "name": "avg_shoe_size_m",
                            "type": "N",
                            "possible_values": None,
                        },
                    ],
                }
            },
        }

        # country as catagorical variable experiment parameters
        tmp = deepcopy(cls.experiment_specific_params_country_as_id_list)
        tmp["features"]["country"] = {
            "type": "C",
            "possible_values": [1, 2, 3, 4, 5],
            "product_set_id": None,
        }
        cls.experiment_specific_params_country_as_categorical = tmp

        # country as dense ID list experiment parameters
        tmp_1 = deepcopy(cls.experiment_specific_params_country_as_id_list)
        tmp_1["features"]["country"] = {
            "type": "P",
            "possible_values": None,
            "product_set_id": "1",
            "use_dense": True,
        }
        cls.experiment_specific_params_country_as_dense_id_list = tmp_1

        cls.shared_params = {
            "data_reader": {},
            "max_epochs": 50,
            "learning_rate": 0.001,
            "l2_decay": 0.0001,
            "batch_size": 64,
            "model": {
                "layers": [-1, 32, 16, -1],
                "activations": ["relu", "relu", "linear"],
            },
            "train_test_split": 0.8,
        }

        # set up datasets - one where "country" is a categorical feature, one
        # where country is an ID list feature, and one where country is an
        # ID list to dense feature.
        raw_data = pd.read_csv(DATASET_PATH)
        offset = int(len(raw_data) * cls.shared_params["train_test_split"])

        # country as categorical dataset
        cls.data_country_categ = train_bandit.preprocess_data(
            raw_data, cls.experiment_specific_params_country_as_categorical
        )
        X, y = train_bandit.data_to_pytorch(
            cls.data_country_categ,
            cls.experiment_specific_params_country_as_categorical["features"],
            cls.experiment_specific_params_country_as_categorical["product_sets"],
        )
        cls.X_country_categ = {
            "X_train": {"X_float": X["X_float"][:offset]},
            "y_train": y[:offset],
            "X_test": {"X_float": X["X_float"][offset:]},
            "y_test": y[offset:],
        }

        # country as id list dataset
        cls.data_country_id_list = train_bandit.preprocess_data(
            raw_data, cls.experiment_specific_params_country_as_id_list
        )
        X, y = train_bandit.data_to_pytorch(
            cls.data_country_id_list,
            cls.experiment_specific_params_country_as_id_list["features"],
            cls.experiment_specific_params_country_as_id_list["product_sets"],
        )
        cls.X_country_id_list = {
            "X_train": {
                "X_float": X["X_float"][:offset],
                "X_id_list": X["X_id_list"][:offset],
                "X_id_list_idxs": X["X_id_list_idxs"][:offset],
            },
            "y_train": y[:offset],
            "X_test": {
                "X_float": X["X_float"][offset:],
                "X_id_list": X["X_id_list"][offset:],
                "X_id_list_idxs": X["X_id_list_idxs"][offset:],
            },
            "y_test": y[offset:],
        }

        # country as dense id list dataset
        cls.data_country_dense_id_list = train_bandit.preprocess_data(
            raw_data, cls.experiment_specific_params_country_as_dense_id_list
        )
        X, y = train_bandit.data_to_pytorch(
            cls.data_country_dense_id_list,
            cls.experiment_specific_params_country_as_dense_id_list["features"],
            cls.experiment_specific_params_country_as_dense_id_list["product_sets"],
        )
        cls.X_country_dense_id_list = {
            "X_train": {"X_float": X["X_float"][:offset]},
            "y_train": y[:offset],
            "X_test": {"X_float": X["X_float"][offset:]},
            "y_test": y[offset:],
        }

        # train benchmark models
        cls.results_gbdt = benchmarks.fit_sklearn_gbdt(
            X_train=cls.X_country_categ["X_train"]["X_float"],
            y_train=cls.X_country_categ["y_train"],
            X_test=cls.X_country_categ["X_test"]["X_float"],
            y_test=cls.X_country_categ["y_test"],
            hyperparams=cls.shared_params,
        )

        cls.results_mlp = benchmarks.fit_sklearn_mlp(
            X_train=cls.X_country_categ["X_train"]["X_float"],
            y_train=cls.X_country_categ["y_train"],
            X_test=cls.X_country_categ["X_test"]["X_float"],
            y_test=cls.X_country_categ["y_test"],
            hyperparams=cls.shared_params,
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pytorch_model_country_as_categorical(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=self.experiment_specific_params_country_as_categorical[
                "features"
            ],
            product_sets=self.experiment_specific_params_country_as_categorical[
                "product_sets"
            ],
            float_feature_order=self.data_country_categ["final_float_feature_order"],
            id_feature_order=self.data_country_categ["final_id_feature_order"],
            layers=self.shared_params["model"]["layers"],
            activations=self.shared_params["model"]["activations"],
            input_dim=train_bandit.num_float_dim(self.data_country_categ),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=self.X_country_categ["X_train"],
            y=self.X_country_categ["y_train"],
            hyperparams=self.shared_params,
        )

        train_mse = skorch_net.history[-1]["train_loss"]
        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT
        assert train_mse < self.results_gbdt["mse_train"] * 1.05
        assert test_mse < self.results_gbdt["mse_test"] * 1.05

        # make sure mse is better or close to out of the box MLP
        assert train_mse < self.results_mlp["mse_train"] * 1.05
        assert test_mse < self.results_mlp["mse_test"] * 1.05

    def test_pytorch_model_country_as_id_list(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=self.experiment_specific_params_country_as_id_list[
                "features"
            ],
            product_sets=self.experiment_specific_params_country_as_id_list[
                "product_sets"
            ],
            float_feature_order=self.data_country_id_list["final_float_feature_order"],
            id_feature_order=self.data_country_id_list["final_id_feature_order"],
            layers=self.shared_params["model"]["layers"],
            activations=self.shared_params["model"]["activations"],
            input_dim=train_bandit.num_float_dim(self.data_country_id_list),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=self.X_country_id_list["X_train"],
            y=self.X_country_id_list["y_train"],
            hyperparams=self.shared_params,
        )

        train_mse = skorch_net.history[-1]["train_loss"]
        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT
        assert train_mse < self.results_gbdt["mse_train"] * 1.05
        assert test_mse < self.results_gbdt["mse_test"] * 1.05

        # make sure mse is better or close to out of the box MLP
        assert train_mse < self.results_mlp["mse_train"] * 1.05
        assert test_mse < self.results_mlp["mse_test"] * 1.05

    def test_pytorch_model_country_as_dense_id_list(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=self.experiment_specific_params_country_as_dense_id_list[
                "features"
            ],
            product_sets=self.experiment_specific_params_country_as_dense_id_list[
                "product_sets"
            ],
            float_feature_order=self.data_country_dense_id_list[
                "final_float_feature_order"
            ],
            id_feature_order=self.data_country_dense_id_list["final_id_feature_order"],
            layers=self.shared_params["model"]["layers"],
            activations=self.shared_params["model"]["activations"],
            input_dim=train_bandit.num_float_dim(self.data_country_dense_id_list),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=self.X_country_dense_id_list["X_train"],
            y=self.X_country_dense_id_list["y_train"],
            hyperparams=self.shared_params,
        )

        train_mse = skorch_net.history[-1]["train_loss"]
        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT
        assert train_mse < self.results_gbdt["mse_train"] * 1.05
        assert test_mse < self.results_gbdt["mse_test"] * 1.05

        # make sure mse is better or close to out of the box MLP
        assert train_mse < self.results_mlp["mse_train"] * 1.05
        assert test_mse < self.results_mlp["mse_test"] * 1.05
