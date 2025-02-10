"""
Created on January 4 10:00:00 2024
@author: Siby Plathottam
"""

import csv
from typing import List,Union

import numpy as np
import matplotlib.pyplot as plt
from powerdatapipeline.config.config import RunConfig

def investigate_csv_file(csv_filepath:str):
	"""Iterate through CSV file and find first row elements, columns, and number of rows"""
	
	print(f"Checking CSV file:{csv_filepath}...")
	with open(csv_filepath, 'r', newline='') as csvfile:
			reader = csv.reader(csvfile)				
			first_row = next(reader, None) ## Check if there is at least one row
			
			assert first_row is not None, f"First row cannot be none!"
			first_row_elements = [cell for cell in first_row]
			n_columns = len(first_row_elements)
			#n_rows = len(list(reader)) #Reads everything into memory
			n_rows = sum(1 for row in reader) #Doesn't read everything into memory			
			column_name_present = all(isinstance(cell, str) for cell in first_row)  # Assuming the first row is column names if it contains strings

			if column_name_present:
				print(f"CSV file contains {n_rows} rows and {n_columns} column names:{[cell for cell in first_row][0:10]}(first 10 columns shown)")
			else:
				print(f"CSV file contains {n_rows} rows and {n_columns} first row elements:{[cell for cell in first_row][0:10]}(first 10 columns shown)")
	
	return column_name_present,first_row_elements,n_columns,n_rows

def count_total_rows(file_path:str):
	with open(file_path, 'r', newline='') as csvfile:
		reader = csv.reader(csvfile)            
		n_rows = len(list(reader))
    
	return n_rows
    
def check_csv_file(csv_filepath:str,use_existing_columnnames:bool,columns_original:List[str]):
	"""Open CSV file and check"""

	column_names_found_in_first_row,first_row_elements,n_columns,n_rows = investigate_csv_file(csv_filepath)
	assert n_rows >=2, "CSV file need to have atleast 2 rows"
	
	if use_existing_columnnames:
		assert column_names_found_in_first_row, "If existing columns names are to be used, string column names should be found in CSV file!"
		assert len(first_row_elements) > 0, "Number of elements in row index 0 should be greater than 0 if using existing column names!"
		if columns_original:
			assert first_row_elements == columns_original, "If user is specifying column names, it should matching with column names found in CSV file"
			print(f"User supplied {len(columns_original)} column names match with {len(first_row_elements)} column names found in CSV file!")
		
		columns_expected = first_row_elements		
		print(f"Using following {len(columns_expected)} existing column names:{columns_expected[0:10]}(only 10 columns shown)...")
	else:
		assert n_columns == len(first_row_elements) == len(columns_original), f"Number of columns in CSV file is {n_columns} while number of user supplied columns is {len(columns_original)}"
		
		columns_expected  = columns_original #Use user provided column names
		print(f"Using following {len(columns_expected)} user supplied column names:{columns_expected}")
	
	if column_names_found_in_first_row:
		header = True #If existing column names are found, then header is present
		header_index = 0 #header names is at row index 0 (only support CSV files where header is at 0)
	else:
		header = False #If existing column names are found, then header is present
		header_index = None

	return header_index,header,columns_expected,n_rows

def datetime_s_to_sinecosine_features(datetime_s:float,time_features:List[str],show_plot:bin=True):
	"""Pack the features into a single array."""

	second = 1
	minute = 60
	hour = 60*60
	day = 24*60*60
	year = (365.2425)*day
	sinecosine_timefeatures = []
	print(f"Creating cyclical vectors for followings time features:{time_features} and {len(datetime_s)} time steps...")
	#print(datetime_s)
	if "second" in time_features:
		secondsine = np.sin(datetime_s * (2 * np.pi / second))	
		secondcosine = np.cos(datetime_s * (2 * np.pi / second))
	if "minute" in time_features:
		minutesine = np.sin(datetime_s * (2 * np.pi / minute))		
		minutecosine = np.cos(datetime_s * (2 * np.pi / minute))
		sinecosine_timefeatures.extend([minutesine.values,minutecosine.values])		
	if "hour" in time_features:
		hoursine = np.sin(datetime_s * (2 * np.pi / hour))
		hourcosine = np.cos(datetime_s * (2 * np.pi / hour))
		sinecosine_timefeatures.extend([hoursine.values,hourcosine.values])	
	if "day" in time_features:
		daysine = np.sin(datetime_s * (2 * np.pi / day))
		daycosine = np.cos(datetime_s * (2 * np.pi / day))
		sinecosine_timefeatures.extend([daysine.values,daycosine.values])
	if "year" in time_features:
		yearsine = np.sin(datetime_s * (2 * np.pi / year))
		yearcosine = np.cos(datetime_s * (2 * np.pi / year))
		sinecosine_timefeatures.extend([yearsine.values,yearcosine.values])

	if show_plot:
		if "minute" in time_features:
			plt.plot(minutesine,label="minutesine")
			plt.plot(minutecosine,label="minutecosine")
		if "hour" in time_features:
			plt.plot(hoursine,label="hoursine")
			plt.plot(hourcosine,label="hourcosine")
		if "day" in time_features:
			plt.plot(daysine,label="daysine")
			plt.plot(daycosine,label="daycosine")
		if "year" in time_features:
			plt.plot(yearsine,label="yearsine")
			plt.plot(yearcosine,label="yearcosine")

		plt.legend()
		plt.xlabel('Time')
		plt.title('Time of day signal')
		plt.savefig('sine_cosine_features.png')
		plt.show()
		plt.clf()
	#print(sinecosine_timefeatures.values)
	return np.array(sinecosine_timefeatures).T

def plot_cyclical_time_features(feature_array:np.ndarray,time_features:List[str],plot_name:str):
	
	i = 0
	for time_feature in time_features:
		print(f"Plotting feature:{time_feature}")	
		if time_feature == "minute":			
			plt.plot(feature_array[:,i],label="minutesine")
			plt.plot(feature_array[:,i+1],label="minutecosine")
			i = i + 2
		if time_feature == "hour":			
			plt.plot(feature_array[:,i],label="hoursine")			
			plt.plot(feature_array[:,i+1],label="hourcosine")
			i = i + 2
		if time_feature == "day":
			print(time_feature)
			plt.plot(feature_array[:,i],label="daysine")
			plt.plot(feature_array[:,i+1],label="daycosine")
			i = i + 2
		if time_feature == "year":
			plt.plot(feature_array[:,i],label="yearsine")
			plt.plot(feature_array[:,i+1],label="yearcosine")
			i = i + 2

	plt.legend()
	plt.xlabel('Time')
	plt.title('Time of day signal')
	plt.savefig(plot_name)
	plt.show()
	plt.clf()

def plot_dataset(feature_array:np.ndarray,plot_name:str):
	"""Plot elements in dataset"""

	array_shape = feature_array.shape
	if len(array_shape) == 2:
		n_features = array_shape[-1]
	else:
		raise ValueError(f"Array shape:{array_shape} is not valid!")

	for i in range(n_features):
		print(f"Plotting feature:{i}")	
		plt.plot(feature_array[:,i],label=f"feature_{i}")		

	print(f"Saving plot in {plot_name}...")
	plt.legend()
	plt.xlabel('Sample')
	plt.title('Dataset features')
	plt.savefig(plot_name)
	plt.show()
	plt.clf()

def convert_seconds(seconds:float):
	"""Converts seconds to days, minutes, and seconds"""
	
	days = seconds // 86400
	remaining_seconds = seconds % 86400

	minutes = remaining_seconds // 60
	remaining_seconds = remaining_seconds % 60

	return days, minutes, remaining_seconds
