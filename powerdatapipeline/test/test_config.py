import unittest
from unittest.mock import Mock
from powerdatapipeline import model_registry
from powerdatapipeline.config.config import RunConfig, register_model

test_config_json = {
    'model': {
        'inference': {
            'model_archivepath': 'test_data/test_model.keras',
            'input_featurespace_archivepath': 'test_data/test_input_featurespace.keras',
            'target_featurespace_archivepath': 'test_data/test_target_featurespace.keras'
        },
        'anomalydetection': {
            'reconstruction_error_threshold': {'AphA': 0.1, 'DCA': 0.1}
        }
    },
    'data_pipeline': {
        'transformation': {
            'input_features': [{'features': ['AphA', 'DCA', 'DCV', 'PhVphA']}],
            'target_features': [{'features': ['AphA', 'DCA']}],
            'window_size': 10,
            'n_rows_eval': 1,
            'batch_size': 10,
            'batch_size_eval': 10,
            'n_rows_to_adapt_featurespace': 10

        },
        'extraction': {'column_datetime': 'datetimestamp',
                       'csv_folder': '.'
                       }
    }
}


class TestConfig(unittest.TestCase):

    def test_config_initialization(self):
        inference_config = RunConfig(**test_config_json)
        assert inference_config


class TestRegisterModel(unittest.TestCase):

    # registered model shouldn't be re-registered
    def test_logs_warning_when_identifier_exists(self):
        model_registry['existing_model'] = Mock()

        @register_model('existing_model')
        class DummyModel:
            def compile(self): pass

            def fit(self): pass

            def call(self): pass

    # invalid model should correctly raise error on registration
    def test_raises_error_when_model_is_missing_required_methods(self):
        with self.assertRaises(AssertionError):
            @register_model('invalid_model')
            class InvalidModel:
                def compile(self): pass

                # 'fit' method is missing so model should be invalid
                def call(self): pass

    # valid model should correctly register
    def test_registers_class_to_model_registry(self):
        @register_model('valid_model')
        class ValidModel:
            def compile(self): pass

            def fit(self): pass

            def call(self): pass

        self.assertIn('valid_model', model_registry)
        self.assertIs(model_registry['valid_model'], ValidModel)


if __name__ == '__main__':
    unittest.main()
