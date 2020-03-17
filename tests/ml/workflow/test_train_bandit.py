import unittest


from ml.models import benchmarks
from tests.fixtures import Params, Datasets
from workflows import train_bandit


class TestTrainBandit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        # train benchmark models
        cls.results_gbdt = benchmarks.fit_sklearn_gbdt(
            X_train=Datasets.X_COUNTRY_CATEG["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG["y_test"],
            hyperparams=Params.SHARED_PARAMS,
        )

        cls.results_mlp = benchmarks.fit_sklearn_mlp(
            X_train=Datasets.X_COUNTRY_CATEG["X_train"]["X_float"],
            y_train=Datasets.X_COUNTRY_CATEG["y_train"],
            X_test=Datasets.X_COUNTRY_CATEG["X_test"]["X_float"],
            y_test=Datasets.X_COUNTRY_CATEG["y_test"],
            hyperparams=Params.SHARED_PARAMS,
        )

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pytorch_model_country_as_categorical(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL[
                "features"
            ],
            product_sets=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_CATEG[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG["final_id_feature_order"],
            layers=Params.SHARED_PARAMS["model"]["layers"],
            activations=Params.SHARED_PARAMS["model"]["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_CATEG),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=Datasets.X_COUNTRY_CATEG["X_train"],
            y=Datasets.X_COUNTRY_CATEG["y_train"],
            hyperparams=Params.SHARED_PARAMS,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.1
        assert test_mse < self.results_mlp["mse_test"] * 1.05

    def test_pytorch_model_country_as_id_list(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST[
                "features"
            ],
            product_sets=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_ID_LIST["final_id_feature_order"],
            layers=Params.SHARED_PARAMS["model"]["layers"],
            activations=Params.SHARED_PARAMS["model"]["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_ID_LIST),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=Datasets.X_COUNTRY_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_ID_LIST["y_train"],
            hyperparams=Params.SHARED_PARAMS,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.1
        assert test_mse < self.results_mlp["mse_test"] * 1.05

    def test_pytorch_model_country_as_dense_id_list(self):
        pytorch_net = train_bandit.build_pytorch_net(
            feature_specs=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST[
                "features"
            ],
            product_sets=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "final_id_feature_order"
            ],
            layers=Params.SHARED_PARAMS["model"]["layers"],
            activations=Params.SHARED_PARAMS["model"]["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_DENSE_ID_LIST),
        )

        skorch_net = train_bandit.fit_custom_pytorch_module_w_skorch(
            module=pytorch_net,
            X=Datasets.X_COUNTRY_DENSE_ID_LIST["X_train"],
            y=Datasets.X_COUNTRY_DENSE_ID_LIST["y_train"],
            hyperparams=Params.SHARED_PARAMS,
        )

        test_mse = skorch_net.history[-1]["valid_loss"]

        # make sure mse is better or close to out of the box GBDT & MLP
        # the GBDT doesn't need as much training so make tolerance more forgiving
        assert test_mse < self.results_gbdt["mse_test"] * 1.1
        assert test_mse < self.results_mlp["mse_test"] * 1.05
