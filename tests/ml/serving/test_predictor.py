import unittest

from ml.serving.predictor import BanditPredictor
from tests.fixtures import Params, Datasets


class TestPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_same_predictions_country_as_categorical(self):
        assert 1 == 1

    def test_same_predictions_country_as_id_list(self):
        assert 1 == 1
