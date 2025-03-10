'''
This file will contain all evaluation functions - created 3/10/2025
'''

#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np
from _0_gene import *

def fitness(
	data	:	np.ndarray	=	None,
	returns	:	np.ndarray	=	None,
	method	:	function	=	None,
	sample	:	int			=	-1,
	holdfor	:	int			=	-1
):
	'''
	### info: ###
	 goes here
	### params:
	returns:
	-
	'''
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