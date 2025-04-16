'''
this function takes what was in the csv_augmod.ipynb file and turns it into a function that can be used easily in other python files/functions
'''

import pandas as pd
import numpy as np
import os
import _Feature_Usage
import _VHS_Features
from typing import Literal

def augmod(
	file_name_in	:	str		=	'live/live_clean.csv'
	,file_name_out	:	str		=	'live/live.csv'
	,format_mode	:	Literal['backtest','live','vhs']	=	'backtest'
	,overwrite_mode	:	bool	=	False
	,clip_stochastic:	bool	=	True
):
	'''This function expands a cleaned dataset and returns the exported file's name, with extension included'''
	#bring in clean data to csv
	data = pd.read_csv(file_name_in)
	#augment the data and expand in a pandas dataframe
	if(format_mode!='vhs'):
		full_set = _Feature_Usage.augmod_dataset(data, index_names=['spx'],format_mode=format_mode, clip_stochastic=clip_stochastic)
	else:
		full_set = _VHS_Features.augmod_dataset(data)

	#check if we are allowed to overwrite, and raise error if not
	if(os.path.exists(file_name_out) & (overwrite_mode == False)):
		raise FileExistsError('FATAL: Tried to augmod dataset, but output file exists and overwrite_mode is False.\nChange one of these settings and try again.')
	
	#export the pandas dataframe as csv
	full_set.to_csv(file_name_out, index=False)
	
	return file_name_out