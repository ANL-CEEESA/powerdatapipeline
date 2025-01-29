"""
Created on December 20 11:00:00 2023
@author: Siby Plathottam
"""

import time
import collections
from typing import List,Union,Dict

import keras
import tensorflow as tf
import numpy as np
import pandas as pd

from tqdm import tqdm

def get_tfdataset_element(dataset:tf.data.Dataset):

	return dataset.take(1).as_numpy_iterator().next()						  

def show_tfdataset_element(dataset:tf.data.Dataset,dataset_name:str,show_first_element_only:bool=False):
	"""Show single element in a TF dataset as numpy array"""
	
	print(f"{dataset_name}:")
	print(f"Element spec:{dataset.element_spec}")
	dataset_element = get_tfdataset_element(dataset)
	
	if isinstance(dataset_element,tuple): #For zipped dataset
		for i,dataset_element_ in enumerate(dataset_element):
			print(f"Tuple {i} - Dataset element shape:{dataset_element_.shape}")
			print(f"Tuple {i} - Dataset element:{dataset_element_}")
	elif isinstance(dataset.element_spec, collections.OrderedDict): #For csv dataset
		for feature,data in dataset_element.items():
			print(f"{feature}:{data}")
	else:
		if not show_first_element_only:
			print(f"Dataset element:{dataset_element}")
		else:
			print(f"Dataset first element:{dataset_element[0]}") #Only show first element in the batch

def compare_tfdataset_elements(dataset_dict):
	print(f"Comparing following datasets:{list(dataset_dict.keys())}")
	for dataset_name,dataset in dataset_dict.items():		
		show_tfdataset_element(dataset,dataset_name)	

def show_tfdataset_cardinatlity(dataset:tf.data.Dataset,dataset_name:str):
	"""Show number of elements"""

	print(f"Cardinality for {dataset_name}:{tf.data.experimental.cardinality(dataset).numpy()}")

def benchmark_tfdataset(dataset:tf.data.Dataset, num_epochs:int=2,dataset_name:str=""):
	"""Benchmark a TF dataset"""
	
	print(f"Running performance benchmark on tfdataset:{dataset_name} for {num_epochs} epochs...")
	tic = time.perf_counter()
	for _ in range(num_epochs):
		for _ in tqdm(dataset):
			# Performing a training step
			#time.sleep(0.001)
			pass
	toc = time.perf_counter()
	print(f"Execution time:{(toc-tic)/num_epochs:.3f} s/epoch")

def tfdataset_to_numpyarray(dataset:tf.data.Dataset,n_elements:int=None,concatenate:bool=False,dataset_name:str=""):
	"""Convert tfdataset to numpy array"""
	
	if n_elements:
		print(f"Converting {n_elements} elements of dataset:{dataset_name} to Numpy array....")
		numpyarray_list = list(dataset.take(n_elements).as_numpy_iterator())
	else:
		print(f"Converting all elements of dataset:{dataset_name} to Numpy array....")
		numpyarray_list = list(dataset.as_numpy_iterator()) #Convert all elements
	if concatenate:
		numpyarray = np.concatenate(numpyarray_list, axis=0)
	else:
		numpyarray = np.array(numpyarray_list)
	print(f"Numpy array shape:{numpyarray.shape}")
	
	return numpyarray

def get_normalizer_from_tfdataset(dataset:tf.data.Dataset,features:List[str],n_elements:Union[int,None]=None,skip_normalization:List[str]=[]):
	"""Normalizer"""

	normalizer_means = []
	normalizer_vars = []
	dataset_array = tfdataset_to_numpyarray(dataset,n_elements,dataset_name="dataset_for_adapting_normalizer")
	assert dataset_array.shape[-1] == len(features), "Number of features mismatched."
	print(f"Calcuating means and variances from {dataset_array.shape[0]} samples...")

	for i,input_feature in enumerate(features):
		if input_feature not in skip_normalization:
			print(f"Adding mean and vars for {input_feature}")			
			normalizer_means.append(dataset_array[:,i].mean(axis=0))
			normalizer_vars.append(dataset_array[:,i].var(axis=0))
		else:
			print(f"Adding identities for {input_feature}")
			normalizer_means.append(0.0)
			normalizer_vars.append(1.0)
	
	print(f"Means:{normalizer_means}, Vars:{normalizer_vars}")
	normalizer = keras.layers.Normalization(mean=tuple(normalizer_means), variance=tuple(normalizer_vars))	
	
	check_normalizer(normalizer,dataset)

	return normalizer

def check_normalizer(normalizer,dataset:tf.data.Dataset):
	print("Checking normalizer...")
	data = get_tfdataset_element(dataset)
	print(f"Data shape:{data.shape}")
	print(f"Raw data:{data}")
	print(f"Normalized data:{normalizer(data)}")

def convert_to_datetimestamp(column_date, column_time): # Function to convert date and time blocks to timestamp
    date_strs = [d.decode('utf-8') for d in column_date.numpy()] # Convert date and time arrays into lists of strings
    time_strs = [t.decode('utf-8') for t in column_time.numpy()]    
    
    datetime_strings = [f"{date} {time}" for date, time in zip(date_strs, time_strs)] # Combine each date and time string into a datetime string    
    
    return tf.constant(datetime_strings, dtype=tf.string) # Convert back to a TensorFlow constant
    
def convert_to_datetimestampseconds(column_datetime:str):
    """Function to convert a column of datetime strings to timestamps."""
    
    datetime_strs = [dt.decode('utf-8') for dt in column_datetime.numpy()] # Decode datetime strings from UTF-8
    timestamps = [int(pd.to_datetime(dt).timestamp()) for dt in datetime_strs] # Convert datetime strings to timestamps (seconds since epoch)
    
    return tf.constant(timestamps, dtype=tf.float64)

def add_datetimestamp(record, column_date:str, column_time:str): # Wrapper function for tf.py_function with user-specified columns
    datetimestamp = tf.py_function(func=convert_to_datetimestamp,inp=[record[column_date], record[column_time]],Tout=tf.string)
    record["datetimestamp"] = datetimestamp
    
    return record

def add_datetimestampseconds(record,column_datetime:str): # Wrapper function for tf.py_function with user-specified columns
    datetimestampseconds = tf.py_function(func=convert_to_datetimestampseconds,inp=[record[column_datetime]],Tout=tf.float64)
    record["datetimestampseconds"] = datetimestampseconds
    
    return record

def add_columns_to_csvdataset(dataset:tf.data.Dataset,columns_added:List[str],column_datetime_dict:Dict=None):
    for column_added in columns_added:
        if column_added == "datetimestamp":            
            print(f"Adding column:{column_added}")
            dataset = dataset.map(lambda record: add_datetimestamp(record, column_datetime_dict["column_date"], column_datetime_dict["column_time"])) # Map the add_timestamp function to add timestamp to each record
            column_datetime_dict.update({"column_datetime":"datetimestamp"})
        elif column_added == "datetimestampseconds":            
            print(f"Adding column:{column_added}")
            dataset = dataset.map(lambda record: add_datetimestampseconds(record, column_datetime_dict["column_datetime"])) # Map the add_timestamp function to add timestamp to each record
        else:
            print(f"Adding column:{column_added} is not supported!")
    
    print(f"Dataset after adding columns:{columns_added}")
    for record in dataset.take(2):
        print({key: {'value':record[key].numpy()[0],'dtype':record[key].dtype} for key in record}) #'shape':record[key].shape
    #for record in dataset.take(2): # To verify output
    #    print(record)

    return dataset

def get_interval_dataset(dataset,column_datetime):
    """Convert to a dataset of tuples for adjacent element comparison"""

    previous = dataset.map(lambda x: x[column_datetime]).skip(1)
    current = dataset.map(lambda x: x[column_datetime]).take(dataset.cardinality().numpy() - 1)

    paired = tf.data.Dataset.zip((current, previous)) # Pair consecutive timestamps
    dataset_interval = paired.map(lambda curr, prev: {"interval": curr - prev}) # Compute differences    
    return dataset_interval
     
def check_equality_in_dataset(dataset, column_name:str):
    """Check if all values in a specified column of a tf.data.Dataset are equal."""
    
    column_dataset = dataset.map(lambda x: x[column_name]) # Extract the column dataset    
    
    def reducer(state, value): # Use reduce to check equality across all elements
        return tf.cond(state[0],
                       lambda: (tf.equal(state[1], value), state[1]),
                       lambda: state)

    # Initialize state (all_equal=True, reference_value=first_value)
    initial_value = column_dataset.take(1).get_single_element()
    initial_state = (tf.constant(True), initial_value)
    
    result, _ = column_dataset.reduce(initial_state, reducer) # Reduce the dataset    
    
    return result

def check_time_intervals(batch,column_datetime:str,time_interval_desired:float):
	"""Extract timestamps and compute intervals within the batch"""
	timestamps = batch[column_datetime]
	intervals = timestamps[1:] - timestamps[:-1]
	
	match = tf.reduce_all(tf.equal(intervals, time_interval_desired)) # Check if all intervals are equal to the desired_time_interval
	
	return match

def get_one_hot_encoder_for_string_column(dataset:tf.data.Dataset,string_column:str,sample_data:list):
	"""Create one hot encoder for the string column in a dataset"""
	
	print(f"Finding unique strings in {string_column}...")
	unique_strings = list(set(list(batch[string_column][0].decode("utf-8") for batch in dataset.as_numpy_iterator())))
	print(f"Unique strings:{unique_strings}")
	one_hot_encoder = keras.layers.StringLookup(vocabulary=unique_strings, output_mode='one_hot')
	#layer.adapt(data)
	
	print(f"Converting sample strings:{sample_data} to one hot encoded data:{one_hot_encoder(sample_data)}")

	return one_hot_encoder
