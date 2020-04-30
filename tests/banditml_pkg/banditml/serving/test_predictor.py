from copy import deepcopy
import json
import os
import unittest

import numpy as np

from banditml_pkg.banditml.preprocessing import preprocessor
from banditml_pkg.banditml.serving.predictor import BanditPredictor
from sklearn.utils import shuffle
from tests.fixtures import Params, Datasets
from utils import model_constructors, model_trainers
from workflows import train_bandit


TMP_NET_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test_model.pt"
TMP_CONFIG_PATH = os.path.dirname(os.path.abspath(__file__)) + "/test_model.json"


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tol = 0.001
        cls.tmp_net_path = TMP_NET_PATH
        cls.tmp_config_path = TMP_CONFIG_PATH

        model_type = Params.ML_PARAMS["model_type"]
        cls.model_params = deepcopy(Params.ML_PARAMS["model_params"][model_type])
        cls.model_params["max_epochs"] = 10

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.tmp_net_path)
        os.remove(cls.tmp_config_path)

    def test_same_predictions_country_as_categorical(self):
        raw_data = shuffle(Datasets._raw_data)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = preprocessor.preprocess_data(
            raw_data,
            Params.ML_PARAMS["data_reader"]["reward_function"],
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = preprocessor.data_to_pytorch(data)

        X_COUNTRY_CATEG = {
            "X_train": {"X_float": _X["X_float"][: Datasets._offset]},
            "y_train": _y[: Datasets._offset],
            "X_test": {"X_float": _X["X_float"][Datasets._offset :]},
            "y_test": _y[Datasets._offset :],
        }

        net_spec, pytorch_net = model_constructors.build_pytorch_net(
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
            reward_type=Params.ML_PARAMS["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_CATEG),
        )

        pre_serialized_predictor = BanditPredictor(
            experiment_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            float_feature_order=Datasets.DATA_COUNTRY_CATEG["float_feature_order"],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG["id_feature_order"],
            id_feature_str_to_int_map=Datasets.DATA_COUNTRY_CATEG[
                "id_feature_str_to_int_map"
            ],
            transforms=Datasets.DATA_COUNTRY_CATEG["transforms"],
            imputers=Datasets.DATA_COUNTRY_CATEG["imputers"],
            net=pytorch_net,
            reward_type=Params.ML_PARAMS["reward_type"],
            net_spec=net_spec,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_PARAMS["reward_type"],
            module=pre_serialized_predictor.net,
            X=X_COUNTRY_CATEG["X_train"],
            y=X_COUNTRY_CATEG["y_train"],
            hyperparams=self.model_params,
        )

        pre_serialized_predictor.config_to_file(self.tmp_config_path)
        pre_serialized_predictor.net_to_file(self.tmp_net_path)

        post_serialized_predictor = BanditPredictor.predictor_from_file(
            self.tmp_config_path, self.tmp_net_path
        )

        pre_pred = pre_serialized_predictor.predict(json.loads(test_input.context))
        post_pred = post_serialized_predictor.predict(json.loads(test_input.context))

        assert np.allclose(pre_pred["scores"], post_pred["scores"], self.tol)
        assert pre_pred["ids"] == post_pred["ids"]

    def test_same_predictions_country_as_id_list(self):
        raw_data = shuffle(Datasets._raw_data)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = preprocessor.preprocess_data(
            raw_data,
            Params.ML_PARAMS["data_reader"]["reward_function"],
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = preprocessor.data_to_pytorch(data)
        X_COUNTRY_ID_LIST = {
            "X_train": {
                "X_float": _X["X_float"][: Datasets._offset],
                "X_id_list": _X["X_id_list"][: Datasets._offset],
                "X_id_list_idxs": _X["X_id_list_idxs"][: Datasets._offset],
            },
            "y_train": _y[: Datasets._offset],
            "X_test": {
                "X_float": _X["X_float"][Datasets._offset :],
                "X_id_list": _X["X_id_list"][Datasets._offset :],
                "X_id_list_idxs": _X["X_id_list_idxs"][Datasets._offset :],
            },
            "y_test": _y[Datasets._offset :],
        }

        net_spec, pytorch_net = model_constructors.build_pytorch_net(
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
            reward_type=Params.ML_PARAMS["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_ID_LIST),
        )

        pre_serialized_predictor = BanditPredictor(
            experiment_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_ID_LIST,
            float_feature_order=Datasets.DATA_COUNTRY_ID_LIST["float_feature_order"],
            id_feature_order=Datasets.DATA_COUNTRY_ID_LIST["id_feature_order"],
            id_feature_str_to_int_map=Datasets.DATA_COUNTRY_ID_LIST[
                "id_feature_str_to_int_map"
            ],
            transforms=Datasets.DATA_COUNTRY_ID_LIST["transforms"],
            imputers=Datasets.DATA_COUNTRY_ID_LIST["imputers"],
            net=pytorch_net,
            reward_type=Params.ML_PARAMS["reward_type"],
            net_spec=net_spec,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_PARAMS["reward_type"],
            module=pre_serialized_predictor.net,
            X=X_COUNTRY_ID_LIST["X_train"],
            y=X_COUNTRY_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        pre_serialized_predictor.config_to_file(self.tmp_config_path)
        pre_serialized_predictor.net_to_file(self.tmp_net_path)

        post_serialized_predictor = BanditPredictor.predictor_from_file(
            self.tmp_config_path, self.tmp_net_path
        )

        pre_pred = pre_serialized_predictor.predict(json.loads(test_input.context))
        post_pred = post_serialized_predictor.predict(json.loads(test_input.context))

        assert np.allclose(pre_pred["scores"], post_pred["scores"], self.tol)
        assert pre_pred["ids"] == post_pred["ids"]

    def test_same_predictions_country_as_dense_id_list(self):
        raw_data = shuffle(Datasets._raw_data)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = preprocessor.preprocess_data(
            raw_data,
            Params.ML_PARAMS["data_reader"]["reward_function"],
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = preprocessor.data_to_pytorch(data)
        X_COUNTRY_DENSE_ID_LIST = {
            "X_train": {"X_float": _X["X_float"][: Datasets._offset]},
            "y_train": _y[: Datasets._offset],
            "X_test": {"X_float": _X["X_float"][Datasets._offset :]},
            "y_test": _y[Datasets._offset :],
        }

        net_spec, pytorch_net = model_constructors.build_pytorch_net(
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
            reward_type=Params.ML_PARAMS["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=train_bandit.num_float_dim(Datasets.DATA_COUNTRY_DENSE_ID_LIST),
        )

        pre_serialized_predictor = BanditPredictor(
            experiment_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_DENSE_ID_LIST,
            float_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_DENSE_ID_LIST["id_feature_order"],
            id_feature_str_to_int_map=Datasets.DATA_COUNTRY_DENSE_ID_LIST[
                "id_feature_str_to_int_map"
            ],
            transforms=Datasets.DATA_COUNTRY_DENSE_ID_LIST["transforms"],
            imputers=Datasets.DATA_COUNTRY_DENSE_ID_LIST["imputers"],
            net=pytorch_net,
            reward_type=Params.ML_PARAMS["reward_type"],
            net_spec=net_spec,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_PARAMS["reward_type"],
            module=pre_serialized_predictor.net,
            X=X_COUNTRY_DENSE_ID_LIST["X_train"],
            y=X_COUNTRY_DENSE_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        pre_serialized_predictor.config_to_file(self.tmp_config_path)
        pre_serialized_predictor.net_to_file(self.tmp_net_path)

        post_serialized_predictor = BanditPredictor.predictor_from_file(
            self.tmp_config_path, self.tmp_net_path
        )

        pre_pred = pre_serialized_predictor.predict(json.loads(test_input.context))
        post_pred = post_serialized_predictor.predict(json.loads(test_input.context))

        assert np.allclose(pre_pred["scores"], post_pred["scores"], self.tol)
        assert pre_pred["ids"] == post_pred["ids"]

    def test_same_predictions_country_and_decision_as_id_list(self):
        raw_data = shuffle(Datasets._raw_data)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = preprocessor.preprocess_data(
            raw_data,
            Params.ML_PARAMS["data_reader"]["reward_function"],
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = preprocessor.data_to_pytorch(data)
        X_COUNTRY_AND_DECISION_ID_LIST = {
            "X_train": {
                "X_float": _X["X_float"][: Datasets._offset],
                "X_id_list": _X["X_id_list"][: Datasets._offset],
                "X_id_list_idxs": _X["X_id_list_idxs"][: Datasets._offset],
            },
            "y_train": _y[: Datasets._offset],
            "X_test": {
                "X_float": _X["X_float"][Datasets._offset :],
                "X_id_list": _X["X_id_list"][Datasets._offset :],
                "X_id_list_idxs": _X["X_id_list_idxs"][Datasets._offset :],
            },
            "y_test": _y[Datasets._offset :],
        }

        net_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST[
                "features"
            ],
            product_sets=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST[
                "product_sets"
            ],
            float_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "final_id_feature_order"
            ],
            reward_type=Params.ML_PARAMS["reward_type"],
            layers=self.model_params["layers"],
            activations=self.model_params["activations"],
            input_dim=train_bandit.num_float_dim(
                Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST
            ),
        )

        pre_serialized_predictor = BanditPredictor(
            experiment_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AND_DECISION_AS_ID_LIST,
            float_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "id_feature_order"
            ],
            id_feature_str_to_int_map=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST[
                "id_feature_str_to_int_map"
            ],
            transforms=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST["transforms"],
            imputers=Datasets.DATA_COUNTRY_AND_DECISION_ID_LIST["imputers"],
            net=pytorch_net,
            reward_type=Params.ML_PARAMS["reward_type"],
            net_spec=net_spec,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=Params.ML_PARAMS["reward_type"],
            module=pre_serialized_predictor.net,
            X=X_COUNTRY_AND_DECISION_ID_LIST["X_train"],
            y=X_COUNTRY_AND_DECISION_ID_LIST["y_train"],
            hyperparams=self.model_params,
        )

        pre_serialized_predictor.config_to_file(self.tmp_config_path)
        pre_serialized_predictor.net_to_file(self.tmp_net_path)

        post_serialized_predictor = BanditPredictor.predictor_from_file(
            self.tmp_config_path, self.tmp_net_path
        )

        pre_pred = pre_serialized_predictor.predict(json.loads(test_input.context))
        post_pred = post_serialized_predictor.predict(json.loads(test_input.context))

        assert np.allclose(pre_pred["scores"], post_pred["scores"], self.tol)
        assert pre_pred["ids"] == post_pred["ids"]

    def test_same_predictions_country_as_categorical_binary_reward(self):
        reward_type = "binary"

        raw_data = shuffle(Datasets._raw_data_binary_reward)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = preprocessor.preprocess_data(
            raw_data,
            Params.REWARD_FUNCTION_BINARY,
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = preprocessor.data_to_pytorch(data)
        X_COUNTRY_CATEG_BINARY_REWARD = {
            "X_train": {"X_float": _X["X_float"][: Datasets._offset_binary_reward]},
            "y_train": _y[: Datasets._offset_binary_reward],
            "X_test": {"X_float": _X["X_float"][Datasets._offset_binary_reward :]},
            "y_test": _y[Datasets._offset_binary_reward :],
        }

        net_spec, pytorch_net = model_constructors.build_pytorch_net(
            feature_specs=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL[
                "features"
            ],
            product_sets=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL[
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
            input_dim=train_bandit.num_float_dim(
                Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD
            ),
        )

        pre_serialized_predictor = BanditPredictor(
            experiment_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            float_feature_order=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD[
                "float_feature_order"
            ],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD[
                "id_feature_order"
            ],
            id_feature_str_to_int_map=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD[
                "id_feature_str_to_int_map"
            ],
            transforms=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD["transforms"],
            imputers=Datasets.DATA_COUNTRY_CATEG_BINARY_REWARD["imputers"],
            net=pytorch_net,
            reward_type=reward_type,
            net_spec=net_spec,
        )

        skorch_net = model_trainers.fit_custom_pytorch_module_w_skorch(
            reward_type=reward_type,
            module=pre_serialized_predictor.net,
            X=X_COUNTRY_CATEG_BINARY_REWARD["X_train"],
            y=X_COUNTRY_CATEG_BINARY_REWARD["y_train"],
            hyperparams=self.model_params,
        )

        pre_serialized_predictor.config_to_file(self.tmp_config_path)
        pre_serialized_predictor.net_to_file(self.tmp_net_path)

        post_serialized_predictor = BanditPredictor.predictor_from_file(
            self.tmp_config_path, self.tmp_net_path
        )

        pre_pred = pre_serialized_predictor.predict(json.loads(test_input.context))
        post_pred = post_serialized_predictor.predict(json.loads(test_input.context))

        assert np.allclose(pre_pred["scores"], post_pred["scores"], self.tol)
        assert pre_pred["ids"] == post_pred["ids"]
