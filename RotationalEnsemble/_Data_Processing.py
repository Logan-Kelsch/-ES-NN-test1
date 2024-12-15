'''
-   Data Preprocessing - - Should end in returning split and reformatted datasets (train/val/ind)
-   -   -   This should be as standard as the previous versions, but broken down into functional code for readability and malleability
'''

#import libraries
import pandas as pd
import numpy as np
import traceback
from _Feature_Usage import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from typing import Literal
import gc

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
	,scaler:	Literal['Standard','Robust','MinMaxScaler']= 'Standard'
	,frmt_lstm:	bool	=		False
	,time_steps:int		=		5
):
	
 	#Input validate All variables
	assert 0 <= indp_size < 1, "Independent set size must be of domain [0,1)."
	assert 0 <  test_size < 1, "Test set size must be of domain (0, 1)."
	
 	#Attempt to load in pandas file
	try:
		data=pd.read_csv(file_name)
	except Exception as e:
		#error output and traceback
		print(f'Could not load file ({file_name}). Please check the file name.')
		traceback.print_exc()
		raise

	#set target here
	if(mod_type == 'Regression'):
		data = data.drop(columns=tn_classification())
		data = data.drop(columns=tn_regression_excpetion(target_t))
	else:   ### CLASSIFICATION #################
		data = data.drop(columns=tn_regression())
		data = data.drop(columns=tn_classification_exception(\
			num_class, split_val, target_t))
	
 	#grab list of all indices of samples in good times
	index_keep = np.where((X[:-time_steps, 5] >= t_start) \
     					& (X[:-time_steps, 5] <= t_end))[0]
	 
	#drop any immediately unwanted features
	data = data.drop(columns=return_name_collection())
 
	#printout number of features and the target
	if(verbose==1):
		print(f'# of Features:	{len(data.columns) - 1}')
		print(f'Target:			{data.columns[-1]}\n')
 
	#split data into features and targets
	X = data.iloc[:, :-1].values
	y = data.iloc[:, -1].values
 
	#collect dict of all features of X
	feat_dict = {index: column for index, column in enumerate(data.columns)}
 
	#'data' will no longer be used
	del data
 
	#update class weights.. if necessary? #determining unneeded
	#label encoder implementation? #determining unneeded
 
	#Standarize features
	scaler = StandardScaler() if scaler=='Standard' else\
    		 RobustScaler() if scaler=='Robust' else \
    		 MinMaxScaler(feature_range=(0,1))
	fit_cutter = int(len(X)*(1-indp_size-test_size))
	scaler.fit(X[:fit_cutter])
	X = scaler.transform(X)	#NOTE X IS OVER-WRITTEN HERE #END NOTE

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
  
	#X\y'_lstm' will no longer be used
	del X_lstm, y_lstm
 
	#collect original number of samples
	len_samples = len(X)
 
	#remove samples of data that 
	X, y = X[index_keep, :], y[index_keep]	# NOTE X IS OVER-WRITTEN HERE #END NOTE
 
	#output number of samples dropped
	print(f'{len_samples - len(index_keep)} Samples Dropped.\n')
 
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
  
	#X and y are no longer used
	del X, y
 
	#output shape of X and y
	print('\nX_train shape == {}.'.format(X_train.shape))
	print('y_train shape == {}.'.format(y_train.shape))
	print('X_val shape == {}.'.format(X_val.shape))
	print('y_val shape == {}.'.format(y_val.shape))
	print('X_ind shape == {}.'.format(X_ind.shape))
	print('y_ind shape == {}.\n'.format(y_ind.shape))
 
	#manually collect all garbage this far
	gc.collect()
 
	#return all data splits THEN the feature list: X's, y's, features
	return 	X_train, X_val, X_ind,\
     		np.squeeze(y_train), np.squeeze(y_val), np.squeeze(y_ind), \
         	feat_dict