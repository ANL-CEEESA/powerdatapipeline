"""
Created on January 20 11:00:00 2024
@author: Siby Plathottam
"""

import os
import collections
import time
import datetime
import glob
from typing import List, Union, Tuple

import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from powerdatapipeline.datapipeline.tfdataset import df_to_dictdataset, csv_to_csvdataset, \
    tfdataset_to_windowed_tfdataset, tfdataset_to_batched_tfdataset, csvdataset_to_tfdataset
from powerdatapipeline.datapipeline.tfdataset import tdataset_to_timefeatures_dataset, concatenate_dataset_features, \
    add_normalizer_to_tfdataset, zip_datasets
from powerdatapipeline.datapipeline.tfdataset_utilities import compare_tfdataset_elements, show_tfdataset_element, \
    tfdataset_to_numpyarray, get_normalizer_from_tfdataset, show_tfdataset_cardinatlity, check_normalizer
from powerdatapipeline.datapipeline.tfdataset_utilities import add_columns_to_csvdataset, get_tfdataset_element, \
    check_equality_in_dataset, get_interval_dataset
from powerdatapipeline.datapipeline.tfdataset_resampling import upsample_to_interval, downsample_to_interval
from powerdatapipeline.datapipeline.datapipeline_utilities import plot_cyclical_time_features, plot_dataset
from powerdatapipeline.datapipeline.pandas_utilities import get_df_der
from powerdatapipeline.utilities.utilities import check_if_file_exists, find_files

np.set_printoptions(precision=4, suppress=True)  # Make numpy values easier to read.


def get_dictdataset_from_csv(config_dict: dict, csv_folder: str, csv_file: str, n_rows: int):
    """Take CSV file and return a dict dataset"""

    use_df = config_dict["data_pipeline"]["extraction"]["use_df"]

    filenames = find_files(filepattern=os.path.join(csv_folder, csv_file))
    for filename in filenames:
        check_if_file_exists(filename, "CSV")

    use_existing_columnnames = config_dict["data_pipeline"]["extraction"]["use_existing_columnnames"]  # True #False
    columns_original = config_dict["data_pipeline"]["extraction"][
        "columns_original"]  # ["index","time", "va", "vb", "vc", "ia", "ib", "ic", "Vdc", "Idc"]
    columns_added = config_dict["data_pipeline"]["extraction"]["columns_added"]  # ["datetimestamp","datetimeseconds"]

    columns_selected = config_dict["data_pipeline"]["extraction"]["columns_selected"]
    column_datetime = config_dict["data_pipeline"]["extraction"]["column_datetime"]
    filtered_column = config_dict["data_pipeline"]["extraction"]["filtered_column"]
    filtered_value = config_dict["data_pipeline"]["extraction"]["filtered_value"]
    columns_dropped = config_dict["data_pipeline"]["extraction"]["columns_dropped"]

    show_df = True
    if use_df:
        print("Using pandas dataframe to create TF dataset...")
        df = get_df_der(data_folder=csv_folder, csv_file=csv_file, n_rows=n_rows,
                        use_existing_columnnames=use_existing_columnnames,
                        columns_original=columns_original, columns_added=columns_added, column_datetime=column_datetime,
                        show_df=show_df)  # der_data_sample

        csvdataset = df_to_dictdataset(df, columns_selected)
        csv_columns = columns_selected
    else:
        print("Directly creating dataset from CSV file...")
        csvdataset, n_rows_in_csvdataset = csv_to_csvdataset(csv_filepattern=os.path.join(csv_folder, csv_file),
                                                             columns_original=columns_original,
                                                             columns_selected=columns_selected,
                                                             use_existing_columnnames=use_existing_columnnames,
                                                             n_rows=n_rows)

    if filtered_column:
        csvdataset = filter_out_string(csvdataset, filtered_column, filtered_value)
        show_elements_in_column(csvdataset, filtered_column)

    if columns_dropped:
        csvdataset = drop_columns_from_csvdataset(csvdataset, columns_dropped)

    csv_columns = list(csvdataset.element_spec.keys())  # Find columns
    show_tfdataset_element(csvdataset, "raw_dict_dataset")

    return csvdataset, csv_columns, n_rows_in_csvdataset


def filter_out_string(csvdataset: tf.data.Dataset, selected_column: str, filtered_string: str) -> tf.data.Dataset:
    print(f"Filtering column:{selected_column} with value:{filtered_string}")
    csvdataset = csvdataset.filter(lambda x: keras.ops.equal(x[selected_column][0],
                                                             filtered_string))  # tf.math.equal(x[selected_column][0],filtered_string)

    return csvdataset


def drop_columns_from_csvdataset(csvdataset: tf.data.Dataset, columns_dropped: list[str]) -> tf.data.Dataset:
    print(f"Dropping following columns:{columns_dropped}")
    csvdataset = csvdataset.map(lambda x: {k: v for k, v in x.items() if k not in columns_dropped})
    print(f"New columns:{list(csvdataset.element_spec.keys())}")

    return csvdataset


def show_elements_in_column(csvdataset: tf.data.Dataset, selected_column: str):
    print(f"Showing column:{selected_column}")
    for data in csvdataset.take(5):
        print(data[selected_column])


def extract_selected_features(element, selected_features: list[str]):
    """Function to extract selected features from a dict dataset"""

    return {feature: element[feature] for feature in selected_features}


def get_dataset_from_rawdataset(rawdataset: tf.data.Dataset, dataset_columns: List[str], dataset_name: str,
                                features_numerical: List[str], features_onehot: List[str], features_time: List[str],
                                feature_datetime: str,
                                plot_datasets: bin = True):
    """Create input target dataset"""

    print(f"Building {dataset_name} dataset...")
    n_elements_to_plot = 1000

    dataset_dict = {}

    if features_numerical:
        dataset_numerical, features_numerical_updated = get_dataset_numerical(rawdataset, dataset_columns,
                                                                              features_numerical, dataset_name)
        dataset_dict.update({"numerical_features": dataset_numerical})
    else:
        features_numerical_updated = []

    if features_onehot:  # One hot features are categorical features that must be converted to one hot features
        dataset_onehot_dict, features_onehot_expanded = get_dataset_onehot(rawdataset, dataset_columns, features_onehot,
                                                                           dataset_name)
        dataset_dict.update(dataset_onehot_dict)
    else:
        features_onehot_expanded = []

    if features_time:
        dataset_time, features_time_updated = get_dataset_time(rawdataset, dataset_columns, features_time,
                                                               feature_datetime, dataset_name, plot_datasets)
        dataset_dict.update({"time_features": dataset_time})
    else:
        features_time_updated = []

    for dataset_key, dataset1 in dataset_dict.items():
        if isinstance(dataset1.element_spec, collections.OrderedDict) or isinstance(dataset1.element_spec,
                                                                                    dict):  # For csv dataset
            print(f"Vectorizing dataset:{dataset_key}...")
            if "onehot" in dataset_key:
                one_hot = True
                print(f"Setting one hot to True")
            dataset_temp, dataset_columns = csvdataset_to_tfdataset(dataset1)
            dataset_dict.update({dataset_key: dataset_temp})

    compare_tfdataset_elements(dataset_dict)

    print(
        f"Building dataset with features:{features_numerical_updated + features_onehot_expanded + features_time_updated}")
    if len(dataset_dict) > 1:
        dataset = concatenate_dataset_features(dataset_dict)
    else:
        dataset = dataset_dict[list(dataset_dict.keys())[0]]

    show_tfdataset_cardinatlity(dataset, dataset_name)
    show_tfdataset_element(dataset, dataset_name)

    dataset_array = tfdataset_to_numpyarray(dataset, n_elements_to_plot, dataset_name=dataset_name)
    if plot_datasets:
        plot_dataset(dataset_array, plot_name=f"plots/{dataset_name}_dataset_features.png")

    return dataset, features_numerical_updated, features_onehot_expanded, features_time_updated


def get_normalized_dataset(dataset: tf.data.Dataset, normalizer, dataset_name: str, n_elements_to_plot: int = 100,
                           plot_datasets: bin = False) -> tf.data.Dataset:
    dataset = add_normalizer_to_tfdataset(dataset, normalizer)
    show_tfdataset_element(dataset, f"{dataset_name}_dataset_normalizer")

    dataset_array_normalized = tfdataset_to_numpyarray(dataset, n_elements_to_plot,
                                                       dataset_name=f"{dataset_name}_features_normalized")

    if plot_datasets:
        plot_dataset(dataset_array_normalized, plot_name=f"plots/{dataset_name}_dataset_features_normalized.png")

    return dataset


def get_transformed_dataset(dataset: tf.data.Dataset, window_size: int, batch_size: int,
                            dataset_name: str) -> tf.data.Dataset:
    if window_size > 1:
        dataset = tfdataset_to_windowed_tfdataset(dataset, window_size)
        show_tfdataset_element(dataset, f"{dataset_name}_dataset_windowed")

    dataset = tfdataset_to_batched_tfdataset(dataset, batch_size=batch_size, use_prefetch=True)
    show_tfdataset_element(dataset, f"{dataset_name}_dataset_windowed_batched",
                           show_first_element_only=True)  # Only show first element of batched dataset to reduce logs

    return dataset


def get_dataset_numerical(dataset: tf.data.Dataset, dataset_columns: str, numerical_features: List[str],
                          dataset_name: str):
    print(f"Adding numerical features:{numerical_features}")
    if isinstance(dataset.element_spec, collections.OrderedDict) or isinstance(dataset.element_spec, dict):
        dataset_numerical = dataset.map(lambda example: {feature: example[feature] for feature in numerical_features})
    else:
        numerical_feature_indexes = [dataset_columns.index(numerical_feature) for numerical_feature in
                                     numerical_features]
        print(f"Numerical {dataset_name} feature indexes:{numerical_feature_indexes}")
        dataset_numerical = dataset.map(lambda window: tf.gather(window, numerical_feature_indexes,
                                                                 axis=-1))  # Gather indices for input numberical features

    return dataset_numerical, numerical_features


def get_dataset_time(dataset: tf.data.Dataset, dataset_columns: str, time_features: List[str],
                     datetime_feature_column: str, dataset_name: str, plot_datasets=False, n_elements_to_plot=10):
    datetime_feature_index = dataset_columns.index(
        datetime_feature_column)  # There can only be one datetime feature index
    print(f"Datetime index:{datetime_feature_index}")

    print(f"Adding time features:{time_features}")
    time_features_cyclical = [f"{cyclical}{time_feature}" for time_feature in time_features for cyclical in
                              ["sin", "cos"]]
    dataset_time = tdataset_to_timefeatures_dataset(dataset, datetime_index=datetime_feature_index,
                                                    time_features=time_features)

    if plot_datasets:
        time_features_array = tfdataset_to_numpyarray(dataset_time, n_elements_to_plot,
                                                      dataset_name=f"{dataset_name}_time_features")
        plot_cyclical_time_features(time_features_array, time_features_cyclical,
                                    f"plots/tfdataset_{dataset_name}_time_features.png")

    return dataset_time, time_features_cyclical


def get_dataset_onehot(dataset, dataset_columns, onehot_features, dataset_name):
    print(f"Adding one hot features:{onehot_features}")
    n_categories = 5
    onehot_dataset_dict = {}
    onehot_features_expanded = []
    for onehot_feature in onehot_features:
        if isinstance(dataset.element_spec, collections.OrderedDict) or isinstance(dataset.element_spec, dict):
            dataset_onehot = dataset.map(
                lambda example: {f"{onehot_feature}_onehot": tf.one_hot(example[onehot_feature], depth=n_categories)})
            dataset_onehot = dataset_onehot.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(
                x))  # Flattens the nested datasets into a single dataset
        else:
            onehot_feature_index = dataset_columns.index(onehot_feature)
            print(f"One hot {dataset_name} feature index:{onehot_feature_index}")
            dataset_onehot = dataset.map(lambda window: tf.gather(window, [onehot_feature_index],
                                                                  axis=-1))  # Gather indices for input categorical features
            dataset_onehot = keras.layers.CategoryEncoding(num_tokens=n_categories, output_mode="one_hot")(
                dataset_onehot)  # Convert categorical to one hot

        onehot_features_expanded.extend(
            [f"{onehot_feature}_onehot_{onehotindex}" for onehotindex in range(n_categories)])
        onehot_dataset_dict.update({f"{onehot_feature}_onehot": dataset_onehot})
        print(f"onehot dataset:{dataset_onehot}")

    return onehot_dataset_dict, onehot_features_expanded


def get_dataset_with_selected_features(dataset: tf.data.Dataset, features: List[str]) -> tf.data.Dataset:
    assert isinstance(dataset.element_spec, collections.OrderedDict) or isinstance(dataset.element_spec,
                                                                                   dict), f"{dataset.element_spec} is not valid!"
    print(f"Found features:{list(dataset.element_spec.keys())}")
    print(f"Selecting features:{features} using map...")

    return dataset.map(lambda example: {feature: example[feature] for feature in features})


def get_featurespace_definitions(feature_specs: List[dict]):
    """Create feature space definitions"""

    feature_space_definitions = {}

    feature_specs = [feature_spec for feature_spec in feature_specs if feature_spec["features"]]
    print(f"Found {len(feature_specs)} feature specifications with with features")

    for feature_spec in feature_specs:
        print(f"Creating feature space for feature type:{feature_spec['feature_type']}")
        if feature_spec["feature_type"] == "numerical":
            for column_name in feature_spec["features"]:
                print(
                    f"Creating numerical (float) feature space for {column_name} with output mode:{feature_spec['output_mode']}")
                if feature_spec["output_mode"] == "plain":
                    feature_space_definitions.update(
                        {column_name: keras.utils.FeatureSpace.float()})  # Numerical features not normalized
                elif feature_spec["output_mode"] == "normalized":
                    feature_space_definitions.update(
                        {column_name: keras.utils.FeatureSpace.float_normalized()})  # Numerical features normalized
                elif feature_spec["output_mode"] == "rescaled":
                    feature_space_definitions.update({
                                                         column_name: keras.utils.FeatureSpace.float_rescaled()})  # Numerical features linearly rescaled
                else:
                    raise ValueError(f"{feature_spec['output_mode']} is not valid!")

        elif feature_spec["feature_type"] == "int":
            for column_name in feature_spec["features"]:
                print(
                    f"Creating categorical feature space for {column_name} with output mode:{feature_spec['output_mode']}")
                if feature_spec["output_mode"] == "int":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.integer_categorical(
                        output_mode="int")})  # Categorical feature encoded as string
                if feature_spec["output_mode"] == "one_hot":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.integer_categorical(
                        num_oov_indices=1, output_mode="one_hot")})  # Categorical feature encoded as string
                else:
                    raise ValueError(f"{feature_spec['output_mode']} is not valid!")

        elif feature_spec["feature_type"] == "string":
            for column_name in feature_spec["features"]:
                print(
                    f"Creating string categorical feature space for {column_name} with output mode:{feature_spec['output_mode']}")
                if feature_spec["output_mode"] == "int":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.string_categorical(
                        output_mode="int")})  # Categorical feature encoded as string
                elif feature_spec["output_mode"] == "one_hot":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.string_categorical(
                        num_oov_indices=0, output_mode="one_hot")})  # Categorical feature encoded as string
                else:
                    raise ValueError(f"{feature_spec['output_mode']} is not valid!")

        elif feature_spec["feature_type"] == "datetimestamp_seconds":
            for column_name in feature_spec["features"]:
                print(
                    f"Creating cyclical feature space for {column_name} with output mode:{feature_spec['output_mode']}")
                if feature_spec["output_mode"] == "plain":
                    feature_space_definitions.update(
                        {column_name: keras.utils.FeatureSpace.float()})  # Numerical time feature not normalized
                elif feature_spec["output_mode"] == "cyclical_minute_hour_day":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.feature(
                        preprocessor=keras.layers.Lambda(cyclical_minute_hour_day, dtype=tf.float64), dtype=tf.float64,
                        output_mode="float"),
                                                      })  # minutes,hours, and days in datetime are encoded using cyclical features
                elif feature_spec["output_mode"] == "cyclical_minute":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.feature(
                        preprocessor=keras.layers.Lambda(cyclical_minute, dtype=tf.float64), dtype=tf.float64,
                        output_mode="float"),
                                                      })  # minutes,hours, and days in datetime are encoded using cyclical features
                elif feature_spec["output_mode"] == "cyclical_hour":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.feature(
                        preprocessor=keras.layers.Lambda(cyclical_hour, dtype=tf.float64), dtype=tf.float64,
                        output_mode="float"),
                                                      })  # minutes,hours, and days in datetime are encoded using cyclical features
                elif feature_spec["output_mode"] == "cyclical_day":
                    feature_space_definitions.update({column_name: keras.utils.FeatureSpace.feature(
                        preprocessor=keras.layers.Lambda(cyclical_day, dtype=tf.float64), dtype=tf.float64,
                        output_mode="float"),
                                                      })  # minutes,hours, and days in datetime are encoded using cyclical features
                else:
                    raise ValueError(f"{feature_spec['output_mode']} is not valid!")

        else:
            raise ValueError(f"{column_name} is and undefined column name!")

    feature_space = keras.utils.FeatureSpace(features=feature_space_definitions, output_mode="dict")  # concat
    print(f"Created following feature space:{feature_space.features}")

    return feature_space


def get_featurespace(dataset: tf.data.Dataset, feature_specs: List[dict], n_rows_to_use: int = 1000):
    """Get feature space for tf dataset"""

    feature_space = get_featurespace_definitions(feature_specs)
    print(f"Adapting feature space using {n_rows_to_use} elements...")
    tic = time.perf_counter()
    feature_space.adapt(dataset.take(n_rows_to_use))
    toc = time.perf_counter()
    print(f"Adapting feature space took:{toc - tic:.3f} s")

    return feature_space


def reorder_features(example, desired_order):
    return {key: example[key] for key in desired_order}


def apply_featurespace(dataset: tf.data.Dataset, feature_space) -> tf.data.Dataset:
    """Apply the feature space on a tf dataset"""

    columns_dataset = list(dataset.element_spec.keys())
    assert set(list(dataset.element_spec.keys())) == set(list(
        feature_space.features.keys())), f"Dataset features:{set(list(dataset.element_spec.keys()))} and feature space features:{set(list(feature_space.features.keys()))} should be same!"
    print(f"Applying feature space to raw dataset using map...")
    dataset = dataset.map(feature_space)
    if columns_dataset == list(dataset.element_spec.keys()):
        print(f"Dataset after feature space has same columns order as before:{list(dataset.element_spec.keys())}")
    else:
        print(
            f"Dataset after feature space has different columns order:{list(dataset.element_spec.keys())} -- applying reorder_dataset method using map")
        dataset = dataset.map(lambda record: reorder_features(record, columns_dataset))
        print(f"Reordered dataset columns after applying reorder method:{list(dataset.element_spec.keys())}")
    # print(f"Element spec after applying featurespace:{dataset.element_spec}")
    dataset = dataset.flat_map(
        lambda x: tf.data.Dataset.from_tensor_slices(x))  # Flattens the nested datasets into a single dataset
    # print(type(dataset.element_spec))

    return dataset


def get_train_test_eval_dataset(dataset: tf.data.Dataset, train_fraction: float = 0.8, test_fraction: float = 0.1) -> \
Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Apply the feature space on a tf dataset"""

    eval_fraction = round(1.0 - train_fraction - test_fraction, 4)
    assert train_fraction + test_fraction + eval_fraction == 1.0, "Fractions should sum to 1.0"
    print(f"Splitting dataset into train:{train_fraction:.2f},test:{test_fraction:.3f},and eval:{eval_fraction:.3f}")

    tic = time.perf_counter()
    train_dataset, test_eval_dataset = keras.utils.split_dataset(dataset, left_size=train_fraction)
    if eval_fraction > 0.0:
        test_fraction = test_fraction / (test_fraction + eval_fraction)
        test_dataset, eval_dataset = keras.utils.split_dataset(test_eval_dataset, left_size=test_fraction)
    else:
        print("No evaluation dataset created since eval fraction is 0.0")
        test_dataset = test_eval_dataset
        eval_dataset = None
    toc = time.perf_counter()
    print(f"Splitting took:{toc - tic:.3f} s")

    return train_dataset, test_dataset, eval_dataset


def get_input_target_dataset(config_dict: dict, dataset, dataset_columns: list[str],
                             input_features_numerical: list[str], input_features_onehot: list[str],
                             input_features_time: list[str],
                             target_features_numerical: list[str], target_features_onehot: list[str],
                             target_features_time: list[str],
                             input_normalizer, plot_datasets: bool = False):
    feature_datetime = config_dict["data_pipeline"]["extraction"]["column_datetime"]
    window_size = config_dict["data_pipeline"]["transformation"]["window_size"]  # 7
    batch_size = config_dict["data_pipeline"]["transformation"]["batch_size"]  # 7

    add_input_normalizer = config_dict["data_pipeline"]["transformation"]["add_input_normalizer"]  # 7
    n_elements_to_adapt_normalizer = config_dict["data_pipeline"]["transformation"]["n_rows_to_adapt_featurespace"]  # 7

    input_dataset, input_features_numerical_updated, input_features_onehot_expanded, input_features_time_updated = get_dataset_from_rawdataset(
        dataset, dataset_columns, "input",
        input_features_numerical, input_features_onehot, input_features_time, feature_datetime,
        plot_datasets=plot_datasets)
    input_features = get_combined_featurenames(input_features_numerical_updated, input_features_onehot_expanded,
                                               input_features_time_updated)

    if add_input_normalizer:
        skip_normalization = input_features_onehot_expanded + input_features_time_updated  # Features for which normalization is not done
        if input_normalizer is None:
            input_normalizer = get_normalizer_from_tfdataset(input_dataset, input_features,
                                                             n_elements_to_adapt_normalizer,
                                                             skip_normalization=skip_normalization)
        else:
            print(
                f"Using normalizer with mean:{input_normalizer.mean.numpy()},var:{input_normalizer.variance.numpy()} provided in argument...")
            check_normalizer(input_normalizer, input_dataset)
        input_dataset = get_normalized_dataset(input_dataset, input_normalizer, "input", n_elements_to_plot=100,
                                               plot_datasets=False)
    else:
        input_normalizer = None
    input_dataset = get_transformed_dataset(input_dataset, window_size, batch_size, "input")

    target_dataset, target_features_numerical_updated, target_features_onehot_expanded, target_features_time_updated = get_dataset_from_rawdataset(
        dataset, dataset_columns, "target",
        target_features_numerical, target_features_onehot, target_features_time, feature_datetime,
        plot_datasets=plot_datasets)  # Don't use normalizer for target
    target_features = get_combined_featurenames(target_features_numerical_updated, target_features_onehot_expanded,
                                                target_features_time_updated)

    target_dataset = get_transformed_dataset(target_dataset, window_size, batch_size, "target")

    input_target_dataset = zip_datasets({"input": input_dataset, "target": target_dataset})

    return input_target_dataset, input_features, target_features, input_normalizer


def get_combined_featurenames(input_features_numerical: List[str] = [], input_features_onehot: List[str] = [],
                              input_features_time: List[str] = []):
    """Get combined feature names in correct order"""

    return input_features_numerical + input_features_onehot + input_features_time


def get_final_dataset(dataset: tf.data.Dataset, features: List[str], featurespace, window_size: int, batch_size: int,
                      dataset_name: str, data_type: str = "float64"):
    print(f"Creating final dataset with following features:{features}")
    n_elements_to_plot = 60
    plot_datasets = True
    final_dataset = get_dataset_with_selected_features(dataset, features)
    print(f"Selected {dataset_name} features:{list(final_dataset.element_spec.keys())}")
    if featurespace is not None:
        final_dataset = apply_featurespace(final_dataset, featurespace)
    # for elment in final_dataset.take(1):
    #	print(elment)
    final_dataset, _ = csvdataset_to_tfdataset(final_dataset, data_type=data_type)

    dataset_array = tfdataset_to_numpyarray(final_dataset, n_elements_to_plot, dataset_name=dataset_name)
    if plot_datasets:
        plot_dataset(dataset_array, plot_name=f"plots/{dataset_name}_dataset_features.png")

    final_dataset = get_transformed_dataset(final_dataset, window_size, batch_size, dataset_name)

    return final_dataset


def string_datetime_to_seconds(x):
    """Custom layer to perform sine-cosine transformation."""
    return keras.ops.convert_to_tensor([datetime.datetime.strftime(x, '%Y/%m/%d-%H:%M:%S.%f').timestamp()])


def get_minute_encoding(x):
    return [keras.ops.sin(x * (2 * 3.14 / 60)), keras.ops.cos(x * (2 * 3.14 / 60))]


def get_hour_encoding(x):
    return [keras.ops.sin(x * (2 * 3.14 / (60 * 60))), keras.ops.cos(x * (2 * 3.14 / (60 * 60)))]


def get_day_encoding(x):
    return [keras.ops.sin(x * (2 * 3.14 / (24 * 60 * 60))), keras.ops.cos(x * (2 * 3.14 / (24 * 60 * 60)))]


def datetime_cyclical(x, encoding_type):
    """Custom layer to perform sine-cosine of datetime."""
    encodings = []
    if encoding_type == "minute_hour_day":
        encoding_components = ["minute", "hour", "day"]

    for encoding_component in encoding_components:
        if encoding_component == "minute":
            encodings.extend(get_minute_encoding(x))
        elif encoding_component == "hour":
            encodings.extend(get_hour_encoding(x))
        elif encoding_component == "day":
            encodings.extend(get_day_encoding(x))

    return keras.ops.concatenate(encodings, axis=-1)


def cyclical_minute_hour_day(x):
    """Custom layer to perform sine-cosine of datetime."""
    # return keras.ops.concatenate([keras.ops.sin(x*(2*3.14/60))], axis=-1)
    return keras.ops.concatenate([keras.ops.sin(x * (2 * 3.14 / 60)), keras.ops.cos(x * (2 * 3.14 / 60)),
                                  keras.ops.sin(x * (2 * 3.14 / (60 * 60))), keras.ops.cos(x * (2 * 3.14 / (60 * 60))),
                                  keras.ops.sin(x * (2 * 3.14 / (24 * 60 * 60))),
                                  keras.ops.cos(x * (2 * 3.14 / (24 * 60 * 60)))], axis=-1)


def cyclical_minute(x):
    """Custom layer to perform sine-cosine transformation."""

    return keras.ops.concatenate([keras.ops.sin(x * (2 * 3.14 / 60)), keras.ops.cos(x * (2 * 3.14 / 60))], axis=-1)


def cyclical_hour(x):
    """Custom layer to perform sine-cosine transformation."""

    return keras.ops.concatenate([keras.ops.sin(x * (2 * 3.14 / (60 * 60))), keras.ops.cos(x * (2 * 3.14 / (60 * 60)))],
                                 axis=-1)


def cyclical_day(x):
    """Custom layer to perform sine-cosine transformation."""

    return keras.ops.concatenate(
        [keras.ops.sin(x * (2 * 3.14 / (24 * 60 * 60))), keras.ops.cos(x * (2 * 3.14 / (24 * 60 * 60)))], axis=-1)


def resample_csvdataset(dataset: tf.data.Dataset, column_datetime: str, time_interval_original: int = None,
                        time_interval_desired: int = None, column_plot: str = None, n_original_data=9000, plot=False) -> tf.data.Dataset:
    print(f"Taking {n_original_data} elements from column:{column_plot} of original dataset for plotting...")
    data = list(dataset.take(n_original_data))
    timestamps = [data1[column_datetime].numpy()[0] for data1 in data]
    values = [data1[column_plot].numpy()[0] for data1 in data]

    if time_interval_desired < time_interval_original:
        print(
            f"Upsampling from {time_interval_original} to {time_interval_desired} seconds and displaying first 2 elements...")
        dataset = dataset.flat_map(
            lambda x: upsample_to_interval(x, time_interval=time_interval_desired, column_datetime=column_datetime,
                                           time_span=time_interval_original,
                                           fill_method="repeat"))  # Step 3: Apply resampling with the desired interval (e.g., 120 seconds)
        for record in dataset.take(2):
            print({key: {'value': record[key].numpy()[0], 'dtype': record[key].dtype} for key in
                   record})  # 'shape':record[key].shape

    elif time_interval_desired > time_interval_original:
        print(
            f"Downsampling from {time_interval_original} to {time_interval_desired} seconds and displaying first 2 elements...")
        dataset = downsample_to_interval(dataset, time_interval=time_interval_desired, column_datetime=column_datetime)
        for record in dataset.take(2):
            print({key: {'value': record[key].numpy()[0], 'dtype': record[key].dtype} for key in
                   record})  # 'shape':record[key].shape

    else:
        print(f"No resampling since {time_interval_desired}=={time_interval_original}")

    n_resampled_data = int(n_original_data * (time_interval_original / time_interval_desired))
    print(f"Taking {n_resampled_data} elements from resampled dataset for plotting...")
    data_resampled = list(dataset.take(n_resampled_data))
    timestamps_resampled = [data1[column_datetime].numpy()[0] for data1 in data_resampled]
    values_resampled = [data1[column_plot].numpy()[0] for data1 in data_resampled]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.title(f'Comparison of Original and Resampled Time Series - {column_plot}')
        plt.plot(timestamps, values, label=f'Original Data (n={n_original_data})', marker='o', color='blue', linestyle='',
                 markersize=6)  # Plot original time series
        plt.plot(timestamps_resampled, values_resampled, label=f'Resampled Data (n={n_resampled_data})', marker='x',
                 color='red', linestyle='', markersize=8)  # Plot resampled time series
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return dataset


def add_select_resample_csvdataset(dataset: tf.data.Dataset, columns_added: List[str], columns_selected: List[str],
                                   column_datetime: str, column_datetimedict, time_interval_original: int = None,
                                   time_interval_desired: int = None, column_name: str = None,
                                   resample: bin = False) -> tf.data.Dataset:
    if columns_added:
        dataset = add_columns_to_csvdataset(dataset, columns_added, column_datetimedict)
    else:
        print("No columns to be added...")
    dataset = get_dataset_with_selected_features(dataset, columns_selected)

    dataset_timeinterval = get_interval_dataset(dataset, column_datetime)
    timeinterval = get_tfdataset_element(dataset_timeinterval)["interval"][0]
    print(f"Calculated time interval:{timeinterval}")

    assert -timeinterval == time_interval_original, f"Calculated time interval:{-timeinterval} is not matching expected time interval:{time_interval_original}"
    all_equal = check_equality_in_dataset(dataset_timeinterval, "interval")
    if all_equal.numpy():
        print(f"All time intervals matching:{all_equal.numpy()}")
    else:
        raise ValueError("All time intervals were matching")

    if resample:
        dataset = resample_csvdataset(dataset, column_datetime, time_interval_original, time_interval_desired,
                                      column_name)
    else:
        print("Not resampling")

    print(f"Dataset after feature selection and resampling...")
    for record in dataset.take(2):
        print({key: record[key].numpy()[0] for key in record})

    return dataset
