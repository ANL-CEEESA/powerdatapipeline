import unittest
import tensorflow as tf
import numpy as np
from unittest.mock import MagicMock, patch, call
from powerdatapipeline.datapipeline.datapipeline import get_featurespace, reorder_features, apply_featurespace, \
    get_train_test_eval_dataset


class TestDataSourceMethods(unittest.TestCase):

    def setUp(self):
        self.dataset = tf.data.Dataset.from_tensor_slices({"column": [1, 2, 3]})
        self.feature_space = MagicMock()
        self.feature_specs = [{"name": "column", "type": "numerical"}]

    @patch("powerdatapipeline.datapipeline.datapipeline.get_featurespace_definitions")
    def test_get_featurespace(self, mocked_get_featurespace_definitions):
        mocked_get_featurespace_definitions.return_value = self.feature_space
        get_featurespace(self.dataset, self.feature_specs)
        mocked_get_featurespace_definitions.assert_called_once_with(self.feature_specs)
        self.feature_space.adapt.assert_called_once()

    def test_reorder_features(self):
        example = {"a": 1, "b": 2}
        desired_order = ["b", "a"]
        expected_result = {"b": 2, "a": 1}
        result = reorder_features(example, desired_order)
        self.assertEqual(result, expected_result)

    def test_apply_featurespace(self):
        feature_space = MagicMock()

        def mock_transformation(x):
            x['column'] = np.array([0.0, 0.0, 0.0])
            return x

        feature_space.side_effect = mock_transformation
        feature_space.features = {"column": "spec"}
        result = apply_featurespace(self.dataset, feature_space)
        self.assertIsInstance(result, tf.data.Dataset)

    @patch("keras.utils.split_dataset", return_value=(MagicMock(), MagicMock()))
    def test_get_train_test_eval_dataset(self, mock_split_dataset):
        get_train_test_eval_dataset(self.dataset)
        self.assertEqual(mock_split_dataset.call_count, 2)

if __name__ == '__main__':
    unittest.main()
