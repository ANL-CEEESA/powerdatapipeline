"""
Created on December 20 11:00:00 2023
@author: Siby Plathottam
"""

from typing import List,Union,Dict

import tensorflow as tf
import numpy as np

def upsample_to_interval(batch, time_interval:int,column_datetime:str,time_span:int,fill_method:str="repeat"):  # interval_sec is user-defined
	timestamp_unix = batch[column_datetime][0]
	values = {col: batch[col][0] for col in batch if col != column_datetime}	
	
	start_time = timestamp_unix // time_interval * time_interval # Round to nearest lower interval and generate range based on specified interval
	resampled_intervals = tf.range(start_time, start_time + time_span, delta=time_interval)  # 1-hour span as example
	
	if fill_method == "repeat":
		resampled_data = {col: tf.expand_dims(tf.repeat(values[col], tf.size(resampled_intervals)), axis=1) for col in values}
		resampled_data[column_datetime] = tf.expand_dims(resampled_intervals, axis=1)  # Timestamps with shape (1,)

	#elif fill_method == "linear":
		#resampled_data = {col: tf.expand_dims(tf.concat([tf.linspace(values[col][i], values[col][i + 1], num=2)	for i in range(len(values[col]) - 1)],axis=0,),axis=1,)	for col in values}
		#resampled_data = {col: tf.expand_dims(tf.concat([tf.linspace(values[col][i], values[col][i + 1], num=2)	for i in range(tf.shape(values[col])[0] - 1).numpy()],axis=0,),	axis=1,	) for col in values}
		#resampled_data[column_datetime] = tf.expand_dims(resampled_intervals, axis=1) # Add the resampled timestamps to the data
	
	else:
		raise ValueError(f"Unsupported fill method: {fill_method}")

	return tf.data.Dataset.from_tensor_slices(resampled_data)

def downsample_to_interval(dataset:tf.data.Dataset, time_interval:int, column_datetime:str):
    """Downsample the dataset by filtering records where the timestamp aligns with the specified interval."""

    def filter_fn(record):
        """
        Filters records where the timestamp aligns with the desired interval.
        """
        timestamp = record[column_datetime]
        
        # Check if the timestamp aligns with the interval
        aligned = tf.equal(tf.math.floormod(timestamp, time_interval), 0)
        
        # Ensure the result is a scalar tf.bool tensor
        return tf.squeeze(aligned)

    # Apply filter to keep only records where the timestamp aligns with the interval
    return dataset.filter(filter_fn)