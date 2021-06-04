import unittest

import numpy as np

import torch
from banditml.banditml.models import benchmarks
from banditml.banditml.training import trainer
from banditml.banditml.utils import model_constructors, model_trainers
from tests.fixtures import Datasets, Params


class TestTrainBandit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        model_type = Params.ML_CONFIG["model_type"]
        cls.model_params = Params.ML_CONFIG["model_params"][model_type]

        # train benchmark models for continuous rewards
        cls.results_gbdt = benchmarks.fit_sklearn_gbdt_regression(
            X_train=Datasets.X_COUNTRY_CATEG["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG["y_test"],
            hyperparams=cls.model_params,
        )

        cls.results_rf = benchmarks.fit_sklearn_rf_regression(
            X_train=Datasets.X_COUNTRY_CATEG["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG["y_test"],
            hyperparams=cls.model_params,
        )

        cls.results_mlp = benchmarks.fit_sklearn_mlp_regression(
            X_train=Datasets.X_COUNTRY_CATEG["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG["y_test"],
            hyperparams=cls.model_params,
        )

        # train benchmark models for binary rewards
        cls.results_gbdt_classification = benchmarks.fit_sklearn_gbdt_classification(
            X_train=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["y_test"],
            hyperparams={},
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pytorch_model_country_as_categorical(self):

        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_CATEG[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG["final_id_feature_order"],
            reward_type=Params.ML_CONFIG["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(Datasets.DATA_COUNTRY_CATEG),
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=pytorch_net,
            X=Datasets.X_COUNTRY_CATEG["X_train"],
            y=Datasets.X_COUNTRY_CATEG["y_train"],
            hyperparams=self.model_params,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.15
        assert test_mse < self.results_mlp["mse_test"] * 1.15

    def test_pytorch_model_country_as_id_list(self):
        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AS_ID_LIST[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AS_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_ID_LIST["final_id_feature_order"],
            reward_type=Params.ML_CONFIG["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(Datasets.DATA_COUNTRY_ID_LIST),
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=pytorch_net,
            X=Datasets.X_COUNTRY_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.15
        assert test_mse < self.results_mlp["mse_test"] * 1.15

    def test_pytorch_model_country_as_dense_id_list(self):
        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AS_DENSE_ID_LIST[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AS_DENSE_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "final_id_feature_order"
            ],
            reward_type=Params.ML_CONFIG["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(Datasets.DATA_COUNTRY_DENSE_ID_LIST),
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=pytorch_net,
            X=Datasets.X_COUNTRY_DENSE_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_DENSE_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.15
        assert test_mse < self.results_mlp["mse_test"] * 1.15

    def test_pytorch_model_country_as_id_list_and_decision_as_id_list(self):
        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_id_feature_order"
            ],
            reward_type=Params.ML_CONFIG["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(
                Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST
            ),
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=pytorch_net,
            X=Datasets.X_COUNTRY_AND_DECISION_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_AND_DECISION_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        # also this is learning 2 embedding tables so need more training time
        # to be compeitive
        assert test_mse < self.results_gbdt["mse_test"] * 1.15
        assert test_mse < self.results_mlp["mse_test"] * 1.15

    def test_pytorch_model_country_as_categorical_binary_reward(self):
        reward_type = "binary"

        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AS_CATEGORICAL[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD[
                "final_id_feature_order"
            ],
            reward_type=reward_type,
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(
                Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD
            ),
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=reward_type,
            model=pytorch_net,
            X=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["X_train"],
            y=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["y_train"].squeeze(),
            hyperparams=self.model_params,
        )

        test_acc = skorch_net.history[-1]["valid_acc"]

        # make sure accuracy is better or close to out of the box GBDT.
        # The GBDT doesn't need as much training so make tolerance more forgiving
        assert test_acc > self.results_gbdt_classification["acc_test"] - 0.03

    def test_gbdt_and_random_forest_model_country_as_categorical(self):

        gbdt = model_constructors.build_gbdt(
            reward_type=Params.ML_CONFIG["reward_type"],
            learning_rate=0.1,
            n_estimators=100,
            max_depth=3,
        )

        trained_gbdt_model, training_stats_gbdt = model_trainers.fit_sklearn_model(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=gbdt,
            X=Datasets.X_COUNTRY_CATEG["X_train"],
            y=Datasets.X_COUNTRY_CATEG["y_train"],
        )

        random_forest = model_constructors.build_random_forest(
            reward_type=Params.ML_CONFIG["reward_type"], n_estimators=100, max_depth=3
        )

        trained_rf_model, training_stats_rf = model_trainers.fit_sklearn_model(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=random_forest,
            X=Datasets.X_COUNTRY_CATEG["X_train"],
            y=Datasets.X_COUNTRY_CATEG["y_train"],
        )

        assert training_stats_gbdt["mse_test"] < self.results_gbdt["mse_test"] * 1.05
        assert training_stats_rf["mse_test"] < self.results_rf["mse_test"] * 1.05

    def test_gbdt_and_random_forest_model_country_as_categorical_binary_reward(self):
        reward_type = "binary"

        gbdt = model_constructors.build_gbdt(
            reward_type=reward_type, learning_rate=0.1, n_estimators=100, max_depth=3
        )

        trained_gbdt_model, training_stats_gbdt = model_trainers.fit_sklearn_model(
            reward_type=reward_type,
            model=gbdt,
            X=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["X_train"],
            y=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["y_train"],
        )

        random_forest = model_constructors.build_random_forest(
            reward_type=reward_type, n_estimators=100, max_depth=3
        )

        trained_rf_model, training_stats_rf = model_trainers.fit_sklearn_model(
            reward_type=reward_type,
            model=random_forest,
            X=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["X_train"],
            y=Datasets.X_COUNTRY_CATEG_BINARY_REWARD["y_train"],
        )

        assert (
            training_stats_gbdt["acc_test"]
            > self.results_gbdt_classification["acc_test"] - 0.03
        )
        assert (
            training_stats_rf["acc_test"]
            > self.results_gbdt_classification["acc_test"] - 0.03
        )

    def test_mixture_density_networks_continuous(self):

        model_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST[
                "features"
            ],
            product_sets=Params.FEATURE_CONFIG_COUNTRY_AND_DECISION_AS_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_id_feature_order"
            ],
            reward_type=Params.ML_CONFIG["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=trainer.num_float_dim(
                Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST
            ),
            is_mdn=True,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_CONFIG["reward_type"],
            model=pytorch_net,
            X=Datasets.X_COUNTRY_AND_DECISION_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_AND_DECISION_ID_LIST["y_train"],
            hyperparams=self.model_params,
            model_name="mixture_density_network",
        )

        X0 = Datasets.X_COUNTRY_AND_DECISION_ID_LIST["X_train"]
        preds = skorch_net.predict(X0)
        Y0 = Datasets.X_COUNTRY_AND_DECISION_ID_LIST["y_train"]

        b_size = skorch_net.batch_size
        idx = range(preds.shape[0])
        mu_est = [i for i in idx if i // b_size % 2 == 0]
        var_est = [i for i in idx if i // b_size % 2 == 1]
        mse = np.mean((preds[mu_est].flatten() - Y0.numpy().flatten()) ** 2)

        assert mse < 25
