import logging
import os
from typing import Optional

from pydantic import BaseModel, model_validator, ValidationError
from pydantic import DirectoryPath, ValidationInfo, field_validator
from pydantic import FilePath, ConfigDict

# Automating validation for training/inference config files
from dercybersecurity import model_registry
# In order to allow for model extensions without library code changes


def register_model(identifier):
    if identifier in model_registry:
        logging.warning(f"Warning: {identifier} overrides an existing model identifier.")

    def wrapper(model_class):
        required_methods = ['compile', 'fit', 'call']

        for method in required_methods:
            assert hasattr(model_class,
                           method), f"Registered model '{identifier}' missing required {method}(...) method."
        model_registry[identifier] = model_class
        return model_class

    return wrapper


# Dictionary wrapper in order to retain current indexing scheme
class BaseModelDict(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    def __getitem__(self, key):
        return getattr(self, key)


def one_of_many_validator_generator(fields):
    def one_of_many_validator(cls, values):
        field_values = [values.get(field) for field in fields]
        if not any(field_values):
            raise ValidationError(f'At least one of {{fields}} must be non-optional and valid')
        return values

    return one_of_many_validator


# Recommended way to check relative paths in pydantic since paths are not available at declaration time
def relative_path_validator_generator(base_dir: str):
    def relative_path_validator(cls, v: str, info: ValidationInfo):
        print("Attempting to validate field: ", v)

        if v and base_dir in info.data and os.path.isfile(os.path.join(info.data['csvfolder'], v)):
            raise ValueError(f"File {{v}} does not exist.")

        return v

    return relative_path_validator

class Transformation(BaseModelDict):
    featurespace_cachepath: Optional[FilePath] = None
    input_featurespace_cachepath: Optional[FilePath] = None
    target_featurespace_cachepath: Optional[FilePath] = None
    features: Optional[list[dict]] = None
    input_features: Optional[list[dict]] = None
    target_features: Optional[list[dict]] = None
    window_size: Optional[int] = None
    batch_size: Optional[int] = None
    batch_size_eval: Optional[int] = None
    add_input_normalizer: bool = False
    n_rows_to_adapt_featurespace: Optional[int] = None
    time_interval_desired: Optional[int] = None

    @field_validator('input_featurespace_cachepath', 'target_featurespace_cachepath', mode='before')
    def coerce_optional_to_none(cls, v):
        if v == "":
            return None
        return v
    
    @model_validator(mode="after")
    def check_features_presence(cls, values):
        if not values.features and not values.input_features:
            raise ValueError("At least one of 'features' or 'input_features' must be provided.")
        return values

class Extraction(BaseModelDict):
        csv_folder: DirectoryPath
        csv_folder_generator: Optional[DirectoryPath] = None
        csv_file_train: Optional[str] = None
        csv_file_eval: Optional[str] = None
        csv_file_modified: Optional[str] = None
        csv_file_generator: Optional[str] = None
        use_streaming: bool = False
        streaming_data_source: Optional[str] = None
        use_df: bool = False
        use_existing_columnnames: bool = True
        n_rows: Optional[int] = None
        n_rows_eval: Optional[int] = None
        column_datetime: str
        columns_original: list[str] = []
        columns_added: list = []
        columns_selected: list[str] = []
        columns_corrupted: list[str] = []
        filtered_column: Optional[str] = None
        filtered_value: Optional[str] = None
        columns_dropped: list = []
        n_parallel_calls: int = 1
        time_interval_original:Optional[int] = None
        column_datetimedict:Optional[dict] = {}

        field_validator('csv_file_train', 'csv_file_eval','csv_file_modified', 'csv_file_generator')(relative_path_validator_generator('csv_folder'))
        model_validator(mode='before')(one_of_many_validator_generator(['csv_file_train', 'csv_file_eval', 'csv_file_generator', 'csv_file_modified']))

        @field_validator('csv_file_train', 'csv_file_eval', 'csv_file_modified', 'csv_file_generator', mode='before')
        def coerce_optional_to_none(cls, v):
            if v == "":
                return None
            return v

class DataPipelineConfig(BaseModelDict):

    class Downsampling(BaseModel):
        downsampling_rate: float = 1.0

    extraction: Extraction
    transformation: Optional[Transformation] = None
    downsampling: Downsampling = Downsampling()

class ModelConfig(BaseModelDict):
    class Architecture(BaseModelDict):
        model_type: str

    class TrainingConfig(BaseModelDict):
        n_epochs: int
        convert_to_tflite: bool
        model_identifier: str
        distribution_strategy: str
        save_as_tfsavedmodel: bool

    class InferenceConfig(BaseModelDict):
        model_archivepath: str
        input_featurespace_archivepath: str
        target_featurespace_archivepath: str

    class AnomalyDetection(BaseModelDict):
        reconstruction_error_threshold: dict

    architecture: Optional[Architecture] = None
    training: Optional[TrainingConfig] = None
    inference: Optional[InferenceConfig] = None
    anomalydetection: Optional[AnomalyDetection] = None

    model_validator(mode='before')(one_of_many_validator_generator(['training', 'inference', 'anomalydetection']))


class RunConfig(BaseModelDict):
    data_pipeline: DataPipelineConfig
    model: Optional[ModelConfig] = None
