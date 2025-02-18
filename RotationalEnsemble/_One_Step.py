'''
This file will be used for:
-   taking a newly exported raw csv file OR filtered csv file 
-   expanding it into a full featurespace
-   predicting based off of this set
-   visualizing the provided candles

Late implementation:
-   create clustering of ALL candle patters (time-series)
		and show most comparable charts?
'''

from typing import Literal
import os
import joblib
import pandas as pd
import _Feature_Usage
import numpy as np
import _Time_Ensemble
import _ToS_CSV_Cleaner
import _CSV_Augmod
import _Utility

def e2e_predict(
	master_names	:	list	=	[]
	,csv_type		:	Literal['clean','dirty']	=	'dirty'
	,dirty_file_name:	str		=	''
	,clean_file_name:	str		=	'live/live_clean.csv'
	,overwrite_mode	:	bool	=	False
	,scaler_path	:	str		=	'live/scaler.joblib'
	,format_LSTM	:	bool	=	False
	,visual_window	:	tuple	=	(645, 800)
	,chronos_fusion_kwargs:dict	=	{
		'fusion_method':'mv',
		'vote_var':4
	}
	,strategy		:	str		=	None
	,clip_stochastic:	bool	=	True
):
	'''END TO END prediction function'''
	assert (len(visual_window) == 2), \
		f'FATAL: ASSERT: visual_window parameter represents start and end of candle charts.\nLength must be 2, got {len(visual_window)}'

	#first step, bring in the raw HLOCVT data
	match(csv_type):
		
		case 'dirty':
			#this denotes that the csv is coming in straight from ToS export
			#clean the file, turn into HLCVT format

			#first check if the given output file exists, then check if we are allowed to overwrite
			if(os.path.exists(clean_file_name)):

				if(overwrite_mode):
					#if overwriting is permitted, even with existing file
					_ToS_CSV_Cleaner.set_csv(file_name_in=dirty_file_name, file_name_out=clean_file_name)

				else:
					#if overwriting is not permitted, must cut function here
					raise FileExistsError('FATAL: csv output file exists with desired name.\nPlease change overwrite_mode OR clean_file_name')
				
			else:
				#no overwriting will happen, so this action is always safe
				_ToS_CSV_Cleaner.set_csv(file_name_in=dirty_file_name, file_name_out=clean_file_name)

			clean_csv_name = clean_file_name

		case 'clean':
			#this denotes that the csv is coming in as a cleaned csv (format is HLCVT)
			#this could either be re-running, live backtesting, or by-hand live data appending to csv
			clean_csv_name = clean_file_name

	#next, turn the clean dataset into an expanded featurespace usable csv 
	expanded_csv_name = _CSV_Augmod.augmod(file_name_in=clean_csv_name, format_mode='live',overwrite_mode=overwrite_mode, clip_stochastic=clip_stochastic)

	#preparing to load in the dataset for predictions, ensuring different dependents currently exist
	
	#checking for existance of dataset that was just augmodded above
	if(not os.path.exists(expanded_csv_name)):
		raise FileNotFoundError('FATAL: Augmod Dataset was not found when preparing to make predictions in e2e.')
	
	#checking for scaler that will be needed for bringing the dataset in
	if(not os.path.exists(scaler_path)):
		raise FileNotFoundError(f'FATAL: Scaler was not found at "{scaler_path}". Please relocate saved scaler for loading predictions dataset.')
	
	#load in features
	X = pd.read_csv(expanded_csv_name)
	index_keep = np.where((X.values[:, X.columns.get_loc('ToD')] >= visual_window[0]) \
     					& (X.values[:, X.columns.get_loc('ToD')] <= visual_window[1])	  \
						& (X.values[:, X.columns.get_loc('DoW')] > 0))[0]
	
	X_raw = X.iloc[index_keep, :].values

	#load in scaler
	scaler = joblib.load(scaler_path)

	#transform the features based off of scaler
	X_scaled = scaler.transform(X_raw)

	#adding this for minimizing future organization if decide to implement this later on
	#probably actually will be discovered based off of loaded data somewhere next to models
	if(format_LSTM):
		raise NotImplementedError('FATAL: LSTM formatting was requested in e2e but has not been implemented.')
		#rough sudo code
		#lstm_style = joblib.load(somewhere)
		#X = format_lstm(X, lstm_style)

	#NOTE currently not going to implement any 'time of day' based sample filtering

	#grab collection of master model predictions for each sample
	chronos_array = _Time_Ensemble.chronos_predict(X=X_scaled, master_names=master_names)

	super_death_6000_predictions = _Time_Ensemble.chronos_fusion(master_predictions=chronos_array, **chronos_fusion_kwargs)

	if(strategy!=None):
		print(f'Showing Strategy: {strategy}')

		strategy_visual_kwargs = None

		if(strategy == '00'):
			strategy_visual_kwargs = {
				'signal' : 80,
				'add_chart' : [517]
			}

	_Utility.show_predictions_chart(X_raw=X_raw, predictions=super_death_6000_predictions, t_start=visual_window[0], t_end=visual_window[1], mode='live', **strategy_visual_kwargs)