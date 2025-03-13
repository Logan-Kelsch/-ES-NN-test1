'''
This file will contain all evaluation functions - created 3/10/2025
'''

#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np
from _0_gene import *

def fitness(
	arr_close	:	np.ndarray	=	None,
	arr_low		:	np.ndarray	=	None,
	data		:	np.ndarray	=	None,
	genes		:	list|np.ndarray	=	None,
	method		:	function	=	None,
	hold_for		:	int		=	-1,
	lag_allow : int = -1,
	specific_data	:	str		=	None
):
	'''
	### info: ###
	 goes here
	### params:
	returns:
	-
	'''
	#function bare-bones assertions
	assert data != None, "No data was provided to the fitness function."
	assert genes != None, "No genes were provided to the fitness function."
	assert method != None, "No ground-truth method was provided to the fitness function."
	assert hold_for != -1, "No holdfor length was provided to the fitness function."
	assert lag_allow != -1, "No Lagallow length was provided to the fitness function."
	
	#close and low data specific assertions and array formations
	match(specific_data):
		case None:
			assert arr_close != None, \
				"Data is not specified, but close data was not provided to the fitness function."
			if(method is martin_ratio):
				assert arr_low != None, \
					"Data is not specified, martin ratio is selected, but low data was not provided to the fitness function."
		case "form_519":
			raise notImplementedError("MAKE INDEX VALUE")
			some_index = -1
			arr_close = data[:, some_index]
			arr_low = data[:, some_index]

	#boolean 2d array containing entry/or-not (0|1) for each gene
	gene_presence = []


	gene_presence = np.array(gene_presence)
	
	return 

def profit_factor(
	data	:	np.ndarray	=	None,
	returns	:	np.ndarray	=	None
):
	wins, losses = 0, 0
	for i in returns:
		if(i>0):
			wins+=1
		if(i<0):
			losses+=1
	return round((wins/losses), 4)

def total_return(
	data	:	np.ndarray	=	None,
	returns	:	np.ndarray	=	None
):
	return sum(returns)

def ulcer_index():
	return

def martin_ratio(
	data	:	np.ndarray	=	None,
	returns	:	np.ndarray	=	None
):
	tr = total_return(data, returns)
	return
