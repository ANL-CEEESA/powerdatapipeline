"""
Created on December 29 10:00:00 2023
@author: Siby Plathottam
"""

import os
import argparse
import json
import glob
from powerdatapipeline.config.config import RunConfig

def get_config_dict(config_file:str) -> RunConfig:
	"""Read JSON configuration file"""
	
	assert ".json" in config_file, f"{config_file} should be JSON file!"
	if not os.path.exists(config_file):
		raise ValueError(f"{config_file} is not a valid file!")
	else:
		print(f"Reading following config file:{config_file}")
	
	f = json.load(open(config_file))
	config = RunConfig(**f)

	return config

def read_user_arguments():
	parser=argparse.ArgumentParser()
	parser.add_argument('-c','--config',help='config to be passed to the anomaly detection model training script',default = "dercybersecurity/config/anomaly_detection_config_anl_fronius.json", required=False)
	parser.add_argument('-b','--backend',help='Backend framework to be used for Keras 3. Options:tensorflow,torch,jax',default = "tensorflow", required=False)
	parser.add_argument('-p','--precision',help='Numerical precision for Keras 3. Options:float32,float64',default = "float64", required=False)
	args=parser.parse_args()
	config_file = args.config
	keras3_backend = args.backend
	numerical_precision = args.precision

	return config_file,keras3_backend,numerical_precision

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

