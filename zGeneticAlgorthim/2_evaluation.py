'''
This file will contain all evaluation functions - created 3/10/2025
'''

#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np

def fitness(
	func	:	function	=	None,
	returns	:	list|dict	=	None,
	data	:	np.ndarray	=	None
):
	'''
	### info: ###
	 goes here
	### params:
	returns:
	-	a dict of instance index and a tuple of (instance return, instance max)'''
	return func(returns)

def profit_factor(returns):
	wins, losses = 0, 0
	for i in returns:
		if(i>0):
			wins+=1
		if(i<0):
			losses+=1
	return round((wins/losses), 4)

def total_return(returns):
	return sum(returns)

def ulcer_index():
	return

def martin_ratio(returns):
	tr = total_return(returns)