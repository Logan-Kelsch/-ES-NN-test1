'''
This file will contain all evaluation functions - created 3/10/2025
'''

#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np
from math import sqrt
from operator import attrgetter
from _0_gene import *

def fitness(
	arr_close	:	np.ndarray	=	None,
	arr_low		:	np.ndarray	=	None,
	data		:	np.ndarray	=	None,
	genes		:	list|np.ndarray	=	None,
	#NOTE removing method parameter, as fully inclusive evaluations will be done first.
	#method		:	function	=	None,
	hold_for	:	int			=	-1,
	lag_allow	:	int			=	-1,
	specific_data	:	str		=	None,
	log_normalize	:	bool	=	False
):
	'''
	### info: ###
	 goes here
	### params:
	returns:
	-
	'''
	#function bare-bones assertions
	#assert data != None, "No data was provided to the fitness function."
	assert genes != None, "No genes were provided to the fitness function."
	#NOTE removed method assertion as im going to do full inclusive runs first END#NOTE!!!
	#assert method != None, "No ground-truth method was provided to the fitness function."
	assert hold_for != -1, "No holdfor length was provided to the fitness function."
	assert lag_allow != -1, "No Lagallow length was provided to the fitness function."
	
	#close and low data specific assertions and array formations
	match(specific_data):
		case None:
			assert arr_close != None, \
				"Data is not specified, but close data was not provided to the fitness function."
			#if(method is martin_ratio):
			#	assert arr_low != None, \
			#		"Data is not specified, martin ratio is selected, but low data was not provided to the fitness function."
		case "form_519":
			
			#some_index = -1
			arr_close = data[:, 2]
			arr_low = data[:, 1]

	#boolean 2d array containing entry/or-not (0|1) for each gene
	gene_presence = []
	length = len(data)

	#test all samples in the set, accounting for
	#lag allowance and hold length
	for i in range(length):
		
		if(i < lag_allow | i > length-hold_for):
			#want to avoid usage of these values for safe analysis
			gene_presence.append([0]*len(genes))
		else:
		
			i_presence = []
		
			#check presence of each gene at each sample
			for g, gene in enumerate(genes):
				matches = True
				for p in gene._patterns:
					#if given pattern holds true
					#print(f"p: {type(p)} at {g}")
					#print(f'v1,v2,l1,l2: {p._v1} {p._v2} {p._l1} {p._l2}')
					if(p._op(data[i-p._l1, p._v1],data[i-p._l2, p._v2])):
						pass#pattern matches
					else:
						#this is reached when any pattern does not match
						matches = False
						break
				#check if matches variable held
				if(matches == False):
					i_presence.append(0)
				else:
					i_presence.append(1)
		
		#now have a fully built sample presence
		gene_presence.append(i_presence)
			
	gene_presence = np.array(gene_presence)
	
	returns = []
	kelsch_ratio = []
	
	gp_len = len(gene_presence)
	
	#now going to take gene presence and define
	#returns and custom index values
	for i in range(length):
		
		if(i < lag_allow | i > length-hold_for):
			#want to avoid usage of these values for safe analysis
			returns.append(gene_presence[i]*0)
			kelsch_ratio.append(gene_presence[i]*0)
		else:
		
			#calculate returns
			if(log_normalize):
				ret_local = np.log(arr_close[i+hold_for]/arr_close[i])
			else:
				ret_local = (arr_close[i+hold_for] - arr_close[i])
			returns.append(gene_presence[i]*ret_local)
			
			#calculate index values here if desired
			ki_local = 0
			entry_price = arr_close[i]
			for c in range(1,hold_for+1):
				if(log_normalize):
					ki = entry_price/arr_low[i+c] #>=1 means is below entry
					ki_local+=(np.log(max(ki, 1))**2)
				else:
					ki_local+=((max(entry_price - arr_low[i+c] , 0)) ** 2)
			#taking square root of the mean drawdown squared
			if(log_normalize):
				#is in log space, so is returns, so i dont think np.exp goes here
				ki_local = (sqrt(ki_local/hold_for))
			else:
				ki_local = sqrt(ki_local/hold_for)
			

			if(ki_local != 0):
				#print(f"{ret_local} ++++++ {ki_local}")
				kelsch_ratio_local = (ret_local / (ki_local))
			else:
				kelsch_ratio_local = 100 #force a maximum value

			kelsch_ratio.append(gene_presence[i]*(kelsch_ratio_local))

	returns = np.array(returns)
	kelsch_ratio = np.array(kelsch_ratio)
	
	#This function will by default return the returns and kelsch_index values for each gene
	#these are iterable, along dim0 (by data sample) are gene column local values
	return returns, kelsch_ratio


def associate(
	genes	:	list,
	returns,
	kelsch_ratio,
	log_normalize
):
	#iterate through all genes and associate calculated values and collected arrays
	for gi, gene in enumerate(genes):
		
		#calculate relevant statistics for each gene
		local_profit_factor = profit_factor(returns[:, gi])
		local_avg_return = average_nonzero(returns[:, gi], log_normalize)
		local_avg_kelsch_ratio = average_nonzero(kelsch_ratio[:, gi])

		#update data within the gene for local storage for quick evaluation or recall
		gene.update(
			#lastarray_returns		=	returns[:, gi],
			#lastarray_kelsch_ratio	=	kelsch_ratio[:, gi],
			lastavg_returns			=	local_avg_return,
			lastavg_kelsch_ratio	=	local_avg_kelsch_ratio,
			last_profit_factor		=	local_profit_factor
		)

	#returns updated genes
	return genes

def sort_population(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return']	=	'profit_factor'
):
	'''
	This function sorts a population based on a specific criteria
	'''

	assert (population != None), "Tried to sort a population that came in as None."
	
	#variable used for sorting in sorted function in attribute getter from operator lib
	metric = ""

	#for each type of criteria added
	match(criteria):
		#profit factor
		case 'profit_factor':
			metric = "last_profit_factor"
		#average return
		case 'average_return':
			metric = "lastavg_returns"
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "lastavg_kelsch_ratio"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({criteria})")
		
	sorted_pop = sorted(population, key=attrgetter(metric))

	#return population sorted by specified metric within each gene
	return sorted_pop

def show_best_gene_patterns(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return']	=	'profit_factor',
	fss			:	list	=	None
):
	s_p = sort_population(population,criteria)
	s_p[0].show_patterns(fss)
	print(f"Profit Factor:	{s_p[0]._last_profit_factor}")
	print(f"Average Return:	{s_p[0]._lastavg_returns}")
	print(f"Average KRatio:	{s_p[0]._lastavg_kelsch_ratio}")


def profit_factor(
	returns	:	any	=	None
):

	wins, losses = 0, 0
	for i in returns:
		if(i>0):
			wins+=1
		if(i<0):
			losses+=1
	
	if(losses == 0):
		if(wins>50):
			raise ValueError(f'Perfect strategy detected, 0 losses, >50 wins!!!!!! AHAHAHAHA go check your code bro')
		else:
			return 0

	return round((wins/losses), 4)


def total_return(
	returns	:	np.ndarray	=	None
):
	return sum(returns)


def average_nonzero(
	array	:	np.ndarray	=	None,
	log_normalize:bool		=	True	
):
	#will be traditionally used for averaging returns or ratios
	filtered = array[array != 0]

	if(log_normalize):
		avg = np.exp(np.mean(filtered)) if filtered.size > 0 else 0
		avg = avg-1
	else:
		avg = np.mean(filtered) if filtered.size > 0 else 0

	return avg



def ulcer_index():
	#do something
	#possibly remove this
	return


def martin_ratio(
	data	:	np.ndarray	=	None,
	returns	:	np.ndarray	=	None
):
	#do something
	#possibly remove this
	return


def simple_generational_stat_output(
	population	:	list	=	None,
	metric		:	str		=	None
):
	all_metrics = []

	#for each type of criteria added
	match(metric):
		#profit factor
		case 'profit_factor':
			metric = "_last_profit_factor"
		#average return
		case 'average_return':
			metric = "_lastavg_returns"
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "_lastavg_kelsch_ratio"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({metric})")

	fetch_metric = attrgetter(metric)

	for gene in population:

		all_metrics.append(fetch_metric(gene))

	avg_metric = np.mean(all_metrics)
	top_metric = max(all_metrics)

	return avg_metric, top_metric