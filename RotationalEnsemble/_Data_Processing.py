'''
-   Data Preprocessing - - Should end in returning split and reformatted datasets (train/val/ind)
-   -   -   This should be as standard as the previous versions, but broken down into functional code for readability and malleability
'''

#import libraries
import pandas as pd
import numpy as np
import sys
import traceback
from _Feature_Usage import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
import gc

#function to simplify code visualization (primary for verbose printouts)
def do_nothing():
	pass

'''
	This function takes care of all data pre-processing with
	a provided set of parameters to fit to all regular conditions.
'''
def preprocess_data(
    file_name:	str		=		'betaset_tmp.csv'
    ,indp_size: float	=		0.05
    ,test_size:	float	=		0.15
    ,shfl_splt:	bool	=		True
    ,t_start:	int		=		570
    ,t_end:		int		=		720
    ,mod_type:	Literal['Classification','Regression']= 'Classification'
    ,target_t:	int		=		15
    ,num_class:	int		=		2
    ,split_val:	int		=		5
    ,verbose:	int		=		1	
	,scaler:	Literal['Standard','Robust','MinMaxScaler','None']= 'Standard'
	,frmt_lstm:	bool	=		False
	,time_steps:int		=		5
	,keep_price:bool	=		True
):
	'''
	Here is a cool area to put function info.

	Parameters:
	- frmt_lstm (bool): Optionally transform data into 3D time series data (LSTM).
 	'''
 	#Input validate All variables
	assert 0 <= indp_size < 1, "Independent set size must be of domain [0,1)."
	assert 0 <  test_size < 1, "Test set size must be of domain (0, 1)."
	
	print("Trying to load CSV file into DataFrame...",end='') if verbose else do_nothing()
 	#Attempt to load in pandas file
	try:
		data=pd.read_csv(file_name)
		#variable for later printout
		print(f"Success.\nSize of dataset:\t{sys.getsizeof(data)}")
		#trying to convert all of the float64 to float32 for memory
		#float_cols = data.select_dtypes(include=['float64']).columns
		data = data.astype('float32')
		print(f"Size after reduction:\t{sys.getsizeof(data)}")
	except Exception as e:
		#error output and traceback
		print(f'\nCould not load file ({file_name}). Please check the file name.')
		traceback.print_exc()
		raise

	print("Trying to drop unused targets...",end="") if verbose else do_nothing()
	#set target here
	if(mod_type == 'Regression'):
		data = data.drop(columns=tn_classification())
		data = data.drop(columns=tn_regression_excpetion(target_t))
	else:   ### CLASSIFICATION #################
		data = data.drop(columns=tn_regression())
		data = data.drop(columns=tn_classification_exception(\
			num_class, split_val, target_t))
	
	print("Success.\nTrying to collect indices of wanted times...",end="") if verbose else do_nothing()
 	#grab list of all indices of samples in good times
	index_keep = np.where((data.values[:-time_steps, 5] >= t_start) \
     					& (data.values[:-time_steps, 5] <= t_end))[0]
  
	#drop real price features if requested by 'keep_price' fucntion variable
	if(keep_price==False):
		print("Success.\nTrying to drop price features...",end="") if verbose else do_nothing()
		data = data.drop(columns=return_name_collection())
	
	print("Success...") if verbose else do_nothing()
	#printout number of features and the target
	if(verbose==1):
		print(f'\n# of Samples:	{len(data.index)}')
		print(f'\n# of Features:	{len(data.columns) - 1}')
		print(f'\nTarget:		{data.columns[-1]}\n')
 
	print("Trying to split DataFrame into X and y...",end="") if verbose else do_nothing()
	#split data into features and targets
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values

	print(type(X),type(X[0]),type(X[0][0]))
 
	print("Success.\nTrying to collect all feature names and indices...",end='') if verbose else do_nothing()
	#collect list of all feature subsets as dicts {feature_index:feature_name}
	feat_dict = fnsubset_to_indexdictlist(data.columns, fn_all_subsets(real_prices=keep_price))
 
	print("Success.\nTrying to clean up...",end='') if verbose else do_nothing()
	#'data' will no longer be used
	del data
 
	print("Success.\nTrying to encode y and make class weights...",end='') if verbose else do_nothing()
	#update class weights.. if necessary? #determining unneeded
	#label encoder implementation? #determining unneeded
	print("Failed [NON-FATAL: NOT IMPLEMENTED]" if 1 else "Success.") if verbose else do_nothing()

	if(scaler!='None'):
		print("Trying to standardize all featurespace from training featurespace...",end='') if verbose else do_nothing()
		#Standarize features
		scaler = StandardScaler() if scaler=='Standard' else\
				RobustScaler() if scaler=='Robust' else \
				MinMaxScaler(feature_range=(0,1))
		fit_cutter = int(len(X)*(1-indp_size-test_size))
		scaler.fit(X[:fit_cutter])
		X = scaler.transform(X)	#NOTE X IS OVER-WRITTEN HERE #END NOTE

	print("Success.\nTrying to format data into 3D LSTM (Time Series) data..." if frmt_lstm else "Success.\n",end="") if verbose else do_nothing()
	#write into LSTM format
	if(frmt_lstm):
		X_lstm = []
		for i in range(time_steps, len(X)):
			#collect previous time_steps rows for X
			X_lstm.append(X[i-time_steps:i])  
			#the corresponding y value for the last time step in the sequence
		X_lstm = np.array(X_lstm)
		y_lstm = y[time_steps:]
		y_lstm = np.squeeze(y_lstm)
		X, y = X_lstm, y_lstm	# NOTE X IS OVER-WRITTEN HERE #END NOTE
		print("Success.\nTrying to clean up...",end='') if verbose else do_nothing()
		#X\y'_lstm' will no longer be used
		del X_lstm, y_lstm
		print("Success.") if verbose else do_nothing()

	print("Trying to drop unwanted time-range samples...",end='') if verbose else do_nothing()
	#collect original number of samples
	len_samples = len(X)
	#remove samples of data that
	#index_keep = index_keep[:] 
	X, y = X[index_keep, :], y[index_keep]	# NOTE X IS OVER-WRITTEN HERE #END NOTE
	
	print("Success.") if verbose else do_nothing()
	#output number of samples dropped
	print(f'\t{len_samples - len(index_keep)} Samples Dropped.\n')

	print("Trying to split X and y into Train/Validation/Independent...",end='') if verbose else do_nothing()
	#split data into train validation and independent
	#	split data into trained and not trained
	X_train, X_test, y_train, y_test =\
		train_test_split(X, y, train_size=(1-test_size-indp_size), shuffle=shfl_splt)
    #	calcluate precent of test samples that are for validaiton set
	val_perc = (test_size)/(test_size+indp_size)
    #	split test samples into what is used in validation and what is for
    #	post-model building performance testing
	X_val, X_ind, y_val, y_ind =\
		train_test_split(X_test, y_test, train_size=val_perc, shuffle=shfl_splt)
    #	trim out all overlapping samples in the time step dimension of LSTM layers
    #	the front half of training samples are not overlapping and need no trimming
	if(frmt_lstm):
		X_val = X_val[time_steps:]
		y_val = y_val[time_steps:]
		X_ind = X_ind[time_steps:]
		y_ind = y_ind[time_steps:]

	print("Success.\nTrying to clean up...",end='') if verbose else do_nothing()
	#X and y are no longer used
	del X, y
	
	
	print("Success.") if verbose else do_nothing()
	#output shape of X and y
	if(verbose):
		print('X_train:\t{}.'	.format(X_train.shape))
		print('y_train:\t{}.'	.format(y_train.shape))
		print('X_val:  \t{}.'		.format(X_val.shape))
		print('y_val:  \t{}.'		.format(y_val.shape))
		print('X_ind:  \t{}.'		.format(X_ind.shape))
		print('y_ind:  \t{}.'		.format(y_ind.shape))

	print("Collecting garbage...",end='') if verbose else do_nothing()
	#manually collect all garbage this far
	gc.collect()

	print("Success.\nTerminating.") if verbose else do_nothing()
	#return all data splits THEN the feature list: X's, y's, features
	return 	X_train, X_val, X_ind,\
     		np.squeeze(y_train), np.squeeze(y_val), np.squeeze(y_ind), \
         	feat_dict