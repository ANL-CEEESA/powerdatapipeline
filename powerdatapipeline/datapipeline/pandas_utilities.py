"""
Created on January 4 10:00:00 2024
@author: Siby Plathottam
"""

import os
from typing import List,Union

import pandas as pd

from powerdatapipeline.datapipeline.datapipeline_utilities import convert_seconds

def get_df(data_folder:str,csv_file:str,column_names:List[str]=None,header:int=0,n_rows:int = 1000,n_rows_to_skip=None) -> pd.DataFrame:
	"""Open a CSV file and return dataframe"""

	print(f"Loading {n_rows} rows from CSV into DER dataframe...")
	df = pd.read_csv(os.path.join(data_folder,csv_file),names = column_names,header=header,nrows = n_rows,index_col = 0,skiprows=n_rows_to_skip)

	return df

def get_df_der(data_folder:str,csv_file:str,n_rows:int = 1000,use_existing_columnnames:bool=True,columns_original:Union[List[str],None]=None,columns_added:Union[List[str],None]=None,
			   column_datetime:str="datetimestamp",add_errors:bool=False,show_df=True) -> pd.DataFrame:
	"""Open a CSV file containing DER operational time-series data and return dataframe"""

	header_index,header,column_names,n_rows = check_csv_file(os.path.join(data_folder,csv_file),use_existing_columnnames,columns_original)
	
	df_der = get_df(data_folder,csv_file,column_names,header_index,n_rows)
	
	numeric_columns  = df_der.columns.to_series().apply(pd.to_numeric, errors='coerce').notna()
	numeric_columns_names = df_der.columns[numeric_columns].tolist()
	if numeric_columns_names:
		raise ValueError(f"Numeric column names found:{numeric_columns_names}")
	
	if df_der.index.name:
		df_der = df_der.reset_index(drop=False) #Don't drop index column if it is a named column
	else:
		df_der = df_der.reset_index(drop=True)

	if column_datetime in columns_added:
		assert column_datetime not in df_der.columns, f"datetime column:{column_datetime} already present in dataframe"
		print(f"Adding datetime column:{column_datetime}...")
		df_der.insert(0, column_datetime, pd.date_range(start='2023-01-01', periods=len(df_der), freq='S'))

	if "datetimeseconds" in columns_added:
		print("Adding datetimeseconds..")
		df_der[column_datetime] = pd.to_datetime(df_der[column_datetime], format='%d.%m.%Y %H:%M:%S')
		df_der.insert(1, 'datetimeseconds', datetime_to_seconds(df_der[column_datetime]))
		
	print(f"Number of measurements:{len(df_der)}")
	if add_errors: #Add errors to dataframe
		df_der.loc[1,"vb"] = 1e7
		df_der = df_der.rename(columns={"va": "Va"})

	if show_df:
		print(df_der.head())
	
	return df_der

def datetime_to_seconds(date_time) -> float:
	"""Pack the features into a single array."""

	datetime_s = date_time.map(pd.Timestamp.timestamp)
	
	return datetime_s

def df_to_csv(df:pd.DataFrame,csv_filepath:str,keep_index:bool=False) -> None:
	"""Save dataframe as CSV"""

	print(f"Saving dataframe with {len(df)} rows and following columns:{list(df.columns)} in {csv_filepath}")
	df.to_csv(csv_filepath,index=keep_index)

def check_column_all_nan(df:pd.DataFrame,column_names:List[str]):
	"""Check if all value in column have NaN values"""
	
	for column_name in column_names:
		print(f"Checking if all values in {column_name} are NaN")
		if not df[column_name].isna().all():
			print(f"Found {len(df[df[column_name].notna()])} non NaN values in {column_name}:\n{df[df[column_name].notna()]} ")
			raise ValueError(f"Expected {column_name} to have only NaN values!")

def check_column_all_not_nan(df:pd.DataFrame,column_names:List[str]):
	"""Check if any value is NaN"""
	
	for column_name in column_names:
		print(f"Checking if all values in {column_name} are not NaN")
		if not df[column_name].isna().any():
			print(f"Found following {len(df[df[column_name].isna()])} NaN values in {column_name}:\n{df[df[column_name].isna()]}")
			raise ValueError(f"Expected {column_name} to have only non-NaN values!")

def drop_columns_from_df(df:pd.DataFrame,column_names:List[str]) -> pd.DataFrame:
	"""Drop columns"""
	
	for column_name in column_names:
		print(f"Dropping {column_name}..")
		df = df.drop(column_name, axis=1)

	return df

def find_minutes_in_df(df:pd.DataFrame,datetime_column:str):
	"""Find time in dataframe"""

	print("Finding number of days,minutes, and seconds...")
	time_difference = df[datetime_column].max() - df[datetime_column].min() # Calculate the difference between maximum and minimum datetime values	
	days, minutes, remaining_seconds = convert_seconds(time_difference.total_seconds())
	print(f"Found following: {days:.2f} days, {minutes:.2f} minutes, and {remaining_seconds:.2f} seconds")

def find_df_timeperiod(df:pd.DataFrame,column_datetime:str="datetime"):
	"""Upsample origina time series"""
	
	delta_t_seconds = (df[column_datetime].iloc[-1]-df[column_datetime].iloc[-2]).total_seconds()
	delta_t_minutes = int(delta_t_seconds/60.0)
	print(f"Original time period:{delta_t_seconds} seconds")
	print(f"Original time period:{delta_t_minutes} minutes")

def get_downsampled_df(df_timeseries:pd.DataFrame,column_datetime:str="datetime",downsample_time_period:str="1S"):
	"""Downsample origina time series"""

	find_df_timeperiod(df_timeseries,column_datetime)
	print(f"Columns in downsampled df:{list(df_timeseries.columns)}")
	df_timeseries_downsampled = df_timeseries.set_index(column_datetime)
    
	df_timeseries_downsampled = df_timeseries_downsampled.resample(downsample_time_period).mean()  # '5T' represents 5-minute interval '1S' represents 1 second interval

	df_timeseries_downsampled = df_timeseries_downsampled.reset_index()
	df_timeseries_downsampled.index.name = "index"
	print("After downsampling...")
	find_df_timeperiod(df_timeseries_downsampled,column_datetime)
	
	return df_timeseries_downsampled

def fill_missing_values_in_df(df:pd.DataFrame,columns_to_avoid:list=[]) -> pd.DataFrame:
	"""Fill missing values in fronius solarpvder data"""
	
	for column in df.columns:	# Fill NaN values in each column
		if df[column].isna().any():
			nan_counts = df[column].isna().sum()
			if column not in columns_to_avoid: #use forward fill for all measurements except cumulative measurements like energy
				print(f"Found {nan_counts} NaN values in {column}! --  filling using forward fill strategy")
				df[column] = df[column].ffill() #Forward fill
			else:
				print(f"Found {nan_counts} NaN values in {column}! --  filling using linear interpolation strategy")
				df[column] = df[column].interpolate(method='linear', limit_direction='forward')
			
			#assert df[column].isna().sum() <=3, f"Abnormal number of NaN values:{df[column].isna().sum()} in {column} after filling!"
			if df[column].isna().sum() > 3:
				print(f"Large number of NaN's:{df[column].isna().sum()} in {column} after filling!")
			if df[column].isna().sum() > 1 and column not in columns_to_avoid:				
				print(f"Using backfill to fill {df[column].isna().sum()} NaN values in {column}...")
				df[column] = df[column].interpolate(method='backfill', limit_direction='backward')
				#print(df[column].head())
	
	return df
