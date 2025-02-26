"""
Created on December 29 10:00:00 2023
@author: Siby Plathottam
"""

import os
import argparse
import json
import glob
import re
from powerdatapipeline.config.config import RunConfig

def get_config_dict(config_file:str,validate_config:bool=True) -> RunConfig:
	"""Read JSON configuration file"""
	
	assert ".json" in config_file, f"{config_file} should be JSON file!"
	if not os.path.exists(config_file):
		raise ValueError(f"{config_file} is not a valid file!")
	else:
		print(f"Reading following config file:{config_file}")
	
	f = json.load(open(config_file))
	if validate_config:
		print("Validation of config file is enabled!")
		config = RunConfig(**f) # Validate the configuration
	else:
		print("Validation of config file is disabled!")
		config = f

	return config

def read_user_arguments():
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config to be passed to the anomaly detection model training script',default = "dercybersecurity/config/anomaly_detection_config_anl_fronius.json", required=False)
	parser.add_argument('-b','--backend',help='Backend framework to be used for Keras 3. Options:tensorflow,torch,jax',default = "tensorflow", required=False)
	parser.add_argument('-p','--precision',help='Numerical precision for Keras 3. Options:float32,float64',default = "float64", required=False)
	parser.add_argument('-nvc','--novalidateconfig',help='Disable config file validation',action='store_true')

	args=parser.parse_args()
	config_file = args.config
	keras3_backend = args.backend
	numerical_precision = args.precision
	validate_config_file = not args.novalidateconfig

	return {"config_file":config_file,"keras3_backend":keras3_backend,"numerical_precision":numerical_precision,"validate_config_file":validate_config_file}

def write_json_file(json_object,json_file_name:str="pydantic_errors.json"):	
	with open(json_file_name, "w") as outfile: # Writing to sample.json
		outfile.write(json_object)

def check_if_file_exists(file:str,file_type:str):
	if not os.path.exists(file):
		raise ValueError(f"{file} is not a valid file!")
	else:
		assert file_type.lower() in file, f"Expected {file_type} file but found:{file}"
		print(f"File:{file} exists!")

def find_files(filepattern:str):
	filenames = [filename for filename in glob.glob(filepattern)]
	print(f"Found following file names:{filenames}")
	return filenames

def validation_errors_to_df(validation_errors):
	"""Convert validation error into a dataframe"""

	error_types = [] # Initialize empty lists to store information
	locations = []
	messages = []
	input_values = []
	row_indexes = []
	measurement_types = []

	# Extract information from each validation error
	for error in validation_errors:
		error_types.append(error["type"])
		locations.append(".".join(map(str, error["loc"])))
		row_indexes.append(error["loc"][1])
		measurement_types.append(error["loc"][2])
		messages.append(error["msg"])
		input_values.append(error["input"])

	# Create a Pandas DataFrame
	df_validation_errors = pd.DataFrame({
		"error_type": error_types,
		"location": locations,
		"row_index":row_indexes,
		"measurement_type":measurement_types,
		"message": messages,
		"measurement_value": input_values
	})

	print(df_validation_errors.head())
	df_validation_errors.to_csv("validation_errors.csv")
	return df_validation_errors

def extract_checkpoint_info(checkpoint_filename):
	# Define a regex pattern that captures:
	# window_size, n_input_features, n_target_features, n_rows, and model_type.
	pattern = r"w-(\d+)_f-(\d+)_o-(\d+)_n-(\d+)_([a-zA-Z_]+)_model\.epoch\d+-loss\d+\.\d+\.keras"
	match = re.match(pattern, checkpoint_filename)
	if not match:
		raise ValueError(f"Filename '{checkpoint_filename}' does not match the expected format.")
	window_size, n_input_features, n_target_features, n_rows, model_type = match.groups()
	return {
		"window_size": window_size,
		"n_input_features": n_input_features,
		"n_target_features": n_target_features,
		"n_rows": n_rows,
		"model_type": model_type
	}

def compare_checkpoint_paths(path1, path2):
	# Extract the file names from the paths
	filename1 = os.path.basename(path1)
	filename2 = os.path.basename(path2)

	# Extract the checkpoint parameters from each filename
	info1 = extract_checkpoint_info(filename1)
	info2 = extract_checkpoint_info(filename2)

	# Compare each key parameter; if there's a mismatch, raise a ValueError with details.
	for key in info1:
		if info1[key] != info2[key]:
			raise ValueError(f"Mismatch in '{key}': '{info1[key]}' (from '{filename1}') != '{info2[key]}' (from '{filename2}')")
	
	return True
