"""
Created on December 20 11:00:00 2023
@author: Siby Plathottam
"""

from typing import List,Union

import numpy as np
import tensorflow as tf

from powerdatapipeline.utilities.utilities import find_files
from powerdatapipeline.datapipeline.datapipeline_utilities import check_csv_file

np.set_printoptions(precision=4, suppress=True) # Make numpy values easier to read.

def csv_to_csvdataset(csv_filepattern:str,columns_original:List[str],columns_selected:List[str],use_existing_columnnames:bool,n_rows:int = None) -> tf.data.Dataset:
	"""Create dataset from CSV files"""

	n_rows_total = 0
	print("Loading CSV file to create TF dataset...")
	filenames = find_files(filepattern=csv_filepattern)
	for filename in filenames:
		_,header,column_names,n_rows_counted = check_csv_file(filename,use_existing_columnnames,columns_original)		
		n_rows_total += n_rows_counted
	
	column_defaults = []
	for column in columns_selected:
		if column == "datetimestampseconds" or column == "cotw":
			column_defaults.append(tf.float64)
		elif column == "datetime" or column == "date_block" or column == "time_block":
			column_defaults.append(tf.string)
		else:
			column_defaults.append(tf.float32)
	#column_defaults = None #[tf.float64]*len(columns_selected) #Need all numerical columns to read as float64 datatypes to avoid precision errors for large numbers.string is not supported for now
	
	print(f"Creating csvdataset from {csv_filepattern} containing {n_rows_total} rows and {len(column_names)} columns:{column_names[0:10]} (first 10 columns shown)...")
	dataset = tf.data.experimental.make_csv_dataset(file_pattern=csv_filepattern,batch_size=1,
												    column_defaults = column_defaults,
												    column_names=column_names,select_columns=columns_selected,
												    header=header,num_epochs=1, shuffle=False,num_rows_for_inference=1000)
	print(f"Created CSV dataset after selecting following columns:{columns_selected}")
	print(f"Element spec:{dataset.element_spec}")	
	
	if n_rows:
		if n_rows > n_rows_total:
			print(f"Only {n_rows_total} rows found in CSV file(s), changing n_rows to {n_rows_total}")
			n_rows = n_rows_total

		print(f"Taking {n_rows} rows from CSV file(s) containing {n_rows_total} rows to create CSV dataset...")		
		dataset = dataset.take(n_rows)
		
	return dataset,n_rows

def pack_columns_to_vector_float64(column_names):
	"""Pack the features into a single array."""
	
	features = tf.stack([tf.cast(x,tf.float64) for x in list(column_names.values())], axis=1)
	
	return features

def pack_columns_to_vector_float32(column_names):
	"""Pack the features into a single array."""
	
	features = tf.stack([tf.cast(x,tf.float32) for x in list(column_names.values())], axis=1)
	
	return features

def pack_columns_to_vector(column_names):
	"""Pack the features into a single array."""
	
	features = tf.stack([x for x in list(column_names.values())], axis=1)
	
	return features

def pack_onehot_columns_to_vector_float64(column_names):
	"""Pack the features into a single array."""
	
	features = tf.expand_dims(tf.concat([tf.cast(x,tf.float64) for x in list(column_names.values())], axis=0),axis=0)
	
	return features

def pack_onehot_columns_to_vector_float64_legacy(column_names):
	"""Pack the features into a single array."""
	
	features = tf.stack([tf.cast(x,tf.float64) for x in list(column_names.values())], axis=0)
	
	return features

def csvdataset_to_tfdataset(dataset:tf.data.Dataset,data_type="float64") -> tf.data.Dataset:
	"""Convert csvdataset from tfdataset"""

	dataset_columns = list(dataset.element_spec.keys())
	print(f"Converting dict dataset with {list(dataset.element_spec.keys())} to vectorized dataset of type:{data_type}...")
	print(f"Original element spec:{dataset.element_spec}")
	for data in dataset.take(1):
		print(f"Original dataset element:{data}")
	
	shapes = [spec.shape for spec in dataset.element_spec.values()] # Get the shapes of all elements    
	print(f"Found following shapes:{shapes}")
	
	shape_1d = all(len(shape) == 1 for shape in shapes) # Check if all shapes are 1-dimensional	
	assert shape_1d, "Expected shapes to be 1-dimensional!"

	shape_consistent = all(shape == shapes[0] for shape in shapes) # Check if all shapes are the same
	
	if len(shapes) == 1:
		print("Using converter for single element...")
		if data_type == "float64":
			dataset = dataset.map(lambda x: tf.cast(tf.stack(list(x.values()), axis=0), dtype=tf.float64))			
		else:
			dataset = dataset.map(lambda x: tf.stack(list(x.values()), axis=0))
	elif len(shapes) >= 1:
		if not shape_consistent:
			print(f"Shapes of all elements in element spec are not consistent:{set(shapes)}")
			print("Using converter for elements with inconsistent 1-d shapes...")
			if data_type == "float64":
				dataset = dataset.map(pack_onehot_columns_to_vector_float64)
		else:
			print("Using converter for elements with consistent 1-d shapes...")
			if data_type == "float64":
				dataset = dataset.map(pack_columns_to_vector_float64)
			elif data_type == "float32":
				dataset = dataset.map(pack_columns_to_vector_float32)
			else:
				dataset = dataset.map(pack_columns_to_vector)
	else:
		raise ValueError("Length of shapes should be greater than 0!")

	dataset = dataset.flat_map(lambda x: tf.data.Dataset.from_tensor_slices(x)) #Flattens the nested datasets into a single dataset

	print(f"Converted element spec:{dataset.element_spec}")
	for data in dataset.take(1):
		print(f"Converted dataset element:{data}")
	
	return dataset,dataset_columns

def pack_time_features(element, datetime_index,time_features:List[str]):
    	
	datetime_feature = element[datetime_index] # Extract datetime feature using the provided indice
	sinecosine_timefeatures = []

	for time_feature in time_features:
		if time_feature == "minute":
			sinecosine_timefeatures.extend([tf.math.sin(datetime_feature * (2 * 3.14 / 60)),tf.math.cos(datetime_feature * (2 * 3.14 / 60))])
		if time_feature == "hour":
			sinecosine_timefeatures.extend([tf.math.sin(datetime_feature * (2 * 3.14 / (60*60))),tf.math.cos(datetime_feature * (2 * 3.14 / (60*60)))])
		if time_feature == "day":
			sinecosine_timefeatures.extend([tf.math.sin(datetime_feature * (2 * 3.14 / (24*60*60))),tf.math.cos(datetime_feature * (2 * 3.14 / (24*60*60)))])

	sinecosine_timefeatures = tf.stack(sinecosine_timefeatures,axis=0)  # stack features into tensor
 	
	return sinecosine_timefeatures

def tdataset_to_timefeatures_dataset(dataset,datetime_index:int = 0,time_features:List[str] = ["minute","hour"]):
	"""Create a timefeature dataset from the original dataset"""

	print(f"Creating dataset using datetime with following features:{time_features}")
	partial_pack_time_features = lambda x: pack_time_features(x,datetime_index,time_features) # Create a partial function with the selected_feature_indices fixed
	dataset_timefeatures = dataset.map(partial_pack_time_features) # Apply the partial_map_function to the dataset
	
	return dataset_timefeatures

def concatenate_elements(element1, element2):
	return tf.concat([element1, element2], axis=0)

def concatenate_two_datasets(dataset1,dataset2):
	"""Concatenate two datasets"""

	dataset1 = cast_tfdataset_to_float64(dataset1) #To prevent datatype mismatch errors that occures in certain corner cases
	dataset2 = cast_tfdataset_to_float64(dataset2) #To prevent datatype mismatch errors that occures in certain corner cases
	zipped_dataset = tf.data.Dataset.zip((dataset1, dataset2))
	
	concatenated_dataset = zipped_dataset.map(concatenate_elements) # Apply the concatenate_elements function to each pair in the zipped dataset

	return concatenated_dataset

def zip_datasets(dataset_dict):
	"""Zip dictionary of datasets"""

	dataset_tuple = tuple(dataset_dict.values())	
	zipped_dataset = tf.data.Dataset.zip(dataset_tuple)

	return zipped_dataset

def concatenate_dataset_features(dataset_dict):
	"""Concatenate a dictionary of datasets"""

	print(f"Concatenating following datasets:{list(dataset_dict.keys())}")
	for data in dataset_dict:
		print(f"Element spec:{dataset_dict[data].element_spec}")
	zipped_dataset = zip_datasets(dataset_dict)
	
	concatenated_dataset = zipped_dataset.map(concatenate_elements) # Apply the concatenate_elements function to each pair in the zipped dataset

	return concatenated_dataset

def tfdataset_to_windowed_tfdataset(dataset,window_size:int):
	"""Create windowed dataset from dataframe"""
		
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

	return dataset

def df_to_tfdataset(df,selected_columns:List[str]):
	"""Convert dataframe to a tfdataset"""
	
	print(f"Selecting:{selected_columns} from df to create numeric tfdataset...")
	dataset = tf.data.Dataset.from_tensor_slices(df[selected_columns])
	print(f"Dataset cardinality:{tf.data.experimental.cardinality(dataset)}")
	
	return dataset

def df_to_dictdataset(df,selected_columns:List[str]):
	"""Convert dataframe to a tfdataset"""
	
	print(f"Selecting:{selected_columns} from df to create a dict tfdataset...")	
	dataset = tf.data.Dataset.from_tensor_slices(dict(df[selected_columns]))
	print(f"Dataset cardinality:{tf.data.experimental.cardinality(dataset)}")
	
	return dataset

def cast_tfdataset_to_float32(dataset):
	"""Cast elements to float32 type"""
	
	print(f"Casting dataset elements to float32...")
	dataset = dataset.map(lambda x: tf.cast(x, tf.float32))

	return dataset

def cast_tfdataset_to_float64(dataset:tf.data.Dataset):
	"""Cast elements to float64 type"""
	
	print(f"Casting dataset elements to float64...")
	dataset = dataset.map(lambda x: tf.cast(x, tf.float64))

	return dataset

def add_normalizer_to_tfdataset(dataset:tf.data.Dataset,normalizer):
	"""Add normalizer to tf.data pipeline"""
	
	print(f"Casting dataset elements to float64...")
	dataset = dataset.map(lambda x:normalizer(x))
	
	return dataset

def df_to_windowed_tfdataset(df,window_size:int):
	"""Create windowed dataset from dataframe"""

	dataset = df_to_tfdataset(df)
	dataset = dataset.window(window_size, shift=1, drop_remainder=True)
	dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

	return dataset

def windowed_dataset_v1(df, window_size:int):
	"""Windowed dataset for next step prediction"""
	
	dataset = df_to_windowed_tfdataset(df,window_size)
	#dataset = get_shuffled_tfdataset(dataset,shuffle_buffer=1000)
	dataset = dataset.map(lambda window: (window[:-1], window[-1])) #First n-1 elements are features, and nth element is target
	
	return dataset

def windowed_dataset_v2(df, window_size,shuffle=False):
	dataset = df_to_windowed_tfdataset(df,window_size)	
	#if shuffle:
	#	dataset = get_shuffled_tfdataset(dataset,shuffle_buffer=1000)
		
	dataset = dataset.map(lambda window: (window[:,1:], window[:,0]))
		
	return dataset

def windowed_dataset_to_windowed_dataset_select_features_targets(dataset,feature_indices:List[int],target_indices:List[int]):
	"""Create windowed dataset with selected features and target indices"""

	#dataset = dataset.map(lambda window: (window[:,1:], window[:,0]))
	print(f"Selecting following indexes: features:{feature_indices}, targets:{target_indices}")
	
	#dataset = dataset.map(lambda window: (window[:,features], window[:,targets]))
	dataset = dataset.map(lambda window: (tf.gather(window, feature_indices, axis=1), tf.gather(window, target_indices, axis=1))) #Gather indices
		
	return dataset

def get_shuffled_tfdataset(dataset:tf.data.Dataset,shuffle_buffer:int=1000):
	"""Shuffle the elements of dataset"""

	return dataset.shuffle(shuffle_buffer)

def tfdataset_to_batched_tfdataset(dataset:tf.data.Dataset,batch_size:int=16,use_prefetch:bool=True):
	"""Convert a dataset to a batched dataset"""

	dataset = dataset.batch(batch_size,drop_remainder=True) #Drop remaining data to avoid errors
	if use_prefetch >= 1:
		print("Using prefetch...")
		dataset = dataset.prefetch(tf.data.AUTOTUNE)
	
	return dataset

def change_dataset_datatype(dataset:tf.data.Dataset,datatype:str="float64"):
	if datatype == "float32": #Cast to float32, default datatype is float64
		dataset = cast_tfdataset_to_float32(dataset) #Float32 datasets are encountering errors when converting to cyclical time features
	elif datatype == "float64":
		dataset = cast_tfdataset_to_float64(dataset) #Float64 datasets are encountering errors during concatenation
	else:
		raise ValueError(f"Unsupported data type:{datatype}")
	
	return dataset

def preprocess_batch(batch, dtypes):
    for col_name, dtype in dtypes.items():
        batch[col_name] = tf.cast(batch[col_name], dtype)
    return batch
