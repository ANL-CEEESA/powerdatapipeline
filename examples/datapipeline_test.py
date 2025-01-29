"""
Created on October 25 12:00:00 2024
@author: Siby Plathottam
"""

import os
import sys
from typing import List,Union

baseDir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))	# Add path of home directory e.g.'/home/GitHub/oedi'
print(f"Adding home directory:{baseDir} to path")
sys.path.insert(0, baseDir)

from powerdatapipeline.utilities.utilities import get_config_dict
from powerdatapipeline.datapipeline.datapipeline import get_dictdataset_from_csv,get_featurespace,get_final_dataset,get_train_test_eval_dataset,add_select_resample_csvdataset
from powerdatapipeline.datapipeline.tfdataset_utilities import benchmark_tfdataset

datapipeline_dict = {"der":{"config_file":os.path.join(r"C:\Users\splathottam\Box Sync\GitLab\powerdatapipeline\config","datafusion_config_der.json")},
					 "nodeload":{"config_file":os.path.join(r"C:\Users\splathottam\Box Sync\GitLab\powerdatapipeline\config","datafusion_config_nodeload.json")}}

datapipeline_final_dict = {"der":{"config_file":os.path.join(r"C:\Users\splathottam\Box Sync\GitLab\powerdatapipeline\config","")},
					       "nodeload":{"config_file":os.path.join(r"C:\Users\splathottam\Box Sync\GitLab\powerdatapipeline\config","")}}

prepare_final_dataset = False

for config_key in datapipeline_dict:
	config_dict = get_config_dict(datapipeline_dict[config_key]["config_file"])
	csv_folder = config_dict["data_pipeline"]["extraction"]["csv_folder"]
	csv_file = config_dict["data_pipeline"]["extraction"]["csv_file_train"]
	n_rows = config_dict["data_pipeline"]["extraction"]["n_rows"]
	column_datetime = config_dict["data_pipeline"]["extraction"]["column_datetime"]
	columns_added = config_dict["data_pipeline"]["extraction"]["columns_added"]
	column_datetimedict = config_dict["data_pipeline"]["extraction"]["column_datetimedict"]

	time_interval_original = config_dict["data_pipeline"]["extraction"]["time_interval_original"]  # User-defined interval in seconds
	feature_specs = config_dict["data_pipeline"]["transformation"]["features"]
	time_interval_desired = config_dict["data_pipeline"]["transformation"]["time_interval_desired"]	 # User-defined interval in seconds	
	n_rows_to_adapt_featurespace = config_dict["data_pipeline"]["transformation"]["n_rows_to_adapt_featurespace"] 

	features = [input_feature for input_feature_specs in feature_specs for input_feature in input_feature_specs["features"]]
	
	dataset,csv_columns,n_rows = get_dictdataset_from_csv(config_dict,csv_folder,csv_file,n_rows)

	if config_key == "der":
		dataset = dataset.skip(45000)

	dataset = add_select_resample_csvdataset(dataset,columns_added,features,column_datetime,column_datetimedict,time_interval_original,time_interval_desired,features[0],resample=True)	
	dataset_featurespace = get_featurespace(dataset, feature_specs,n_rows_to_use=n_rows_to_adapt_featurespace)  # Get feature space for input features	
	datapipeline_dict[config_key].update({"dataset":dataset})
	datapipeline_dict[config_key].update({"dataset_featurespace":dataset_featurespace})	

for config_key in datapipeline_dict:
	dataset = datapipeline_dict[config_key]["dataset"]
	dataset_featurespace = datapipeline_dict[config_key]["dataset_featurespace"]
	print(f"Showing elements of dataset:{config_key}...")
	for x in dataset.take(1):	 
		preprocessed_x = dataset_featurespace(x)
		print(f"x sample: {x}")
		print(f"preprocessed_x sample: \n{preprocessed_x}")

	benchmark_tfdataset(dataset,num_epochs=2,dataset_name=f"dataset_{config_key}")

if prepare_final_dataset:
	for config_key in datapipeline_final_dict:
		print(f"Prepare final dataset for:{config_key} using configuration in {datapipeline_final_dict[config_key]['config_file']}...")
		config_dict = get_config_dict(datapipeline_final_dict[config_key]["config_file"])

		input_feature_specs = config_dict["data_pipeline"]["transformation"]["input_features"]
		target_feature_specs = config_dict["data_pipeline"]["transformation"]["target_features"]
		window_size = config_dict["data_pipeline"]["transformation"]["time_interval_desired"]	 # User-defined interval in seconds
		batch_size = config_dict["data_pipeline"]["transformation"]["batch_size"]	 # User-defined interval in seconds

		input_features = [input_feature for input_feature_specs in input_feature_specs for input_feature in input_feature_specs["features"]]
		target_features = [target_feature for target_feature_specs in target_feature_specs for target_feature in target_feature_specs["features"]]

		dataset = datapipeline_dict[config_key]["dataset"]
		dataset_featurespace = datapipeline_dict[config_key]["dataset_featurespace"]
		
		dataset_train, dataset_test, _ = get_train_test_eval_dataset(dataset, train_fraction=0.8,test_fraction=0.2)	 # Splitting dataset into train + test
		input_dataset_train = get_final_dataset(dataset_train, input_features, dataset_featurespace, window_size, batch_size,dataset_name="input_train",data_type = "float64")
		input_dataset_test = get_final_dataset(dataset_test, input_features, dataset_featurespace, window_size, batch_size,dataset_name="input_test",data_type = "float64")
