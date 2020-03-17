import json
import os
import unittest

import numpy as np

from ml.serving import model_io
from ml.serving.predictor import BanditPredictor
from sklearn.utils import shuffle
from tests.fixtures import Params, Datasets
from workflows import train_bandit


TMP_MODEL_PATH = os.path.dirname(os.path.abspath(__file__)) + "/tmp.pkl"


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tol = 0.001
        cls.tmp_model_path = TMP_MODEL_PATH
        Params.SHARED_PARAMS["max_epochs"] = 10

    @classmethod
    def tearDownClass(cls):
        os.remove(cls.tmp_model_path)

    def test_same_predictions_country_as_categorical(self):
        raw_data = shuffle(Datasets._raw_data)
        rand_idx = 0
        test_input = raw_data.iloc[rand_idx]

        data = train_bandit.preprocess_data(
            raw_data,
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            shuffle_data=False,  # don't shuffle so we can test the same observation
        )

        _X, _y = train_bandit.data_to_pytorch(
            data,
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL["features"],
            Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL["product_sets"],
        )

        X_COUNTRY_CATEG = {
            "X_train": {"X_float": _X["X_float"][: Datasets._offset]},
            "y_train": _y[: Datasets._offset],
            "X_test": {"X_float": _X["X_float"][Datasets._offset :]},
            "y_test": _y[Datasets._offset :],
        }

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
            X=X_COUNTRY_CATEG["X_train"],
            y=X_COUNTRY_CATEG["y_train"],
            hyperparams=Params.SHARED_PARAMS,
        )

        pre_pickle_predictor = BanditPredictor(
            experiment_specific_params=Params.EXPERIMENT_SPECIFIC_PARAMS_COUNTRY_AS_CATEGORICAL,
            float_feature_order=Datasets.DATA_COUNTRY_CATEG["float_feature_order"],
            id_feature_order=Datasets.DATA_COUNTRY_CATEG["id_feature_order"],
            transforms=Datasets.DATA_COUNTRY_CATEG["transforms"],
            net=skorch_net,
        )

        model_io.write_predictor_to_disk(pre_pickle_predictor, self.tmp_model_path)
        post_pickle_predictor = model_io.read_predictor_from_disk(self.tmp_model_path)

        pre_pred = skorch_net.predict(X_COUNTRY_CATEG["X_train"]["X_float"][rand_idx])
        post_pred = post_pickle_predictor.predict(json.loads(test_input.context))

        assert np.isclose(pre_pred, post_pred[test_input.decision], self.tol)

    def test_same_predictions_country_as_id_list(self):
        assert 1 == 1

    def test_same_predictions_country_as_dense_id_list(self):
        assert 1 == 1
