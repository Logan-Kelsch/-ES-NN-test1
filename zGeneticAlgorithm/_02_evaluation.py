'''
This file will contain all evaluation functions - created 3/10/2025
'''



#DEV NOTE ensure all fitfuncs are generated parallel to data to allow for parallel analysis.

import numpy as np
from math import sqrt
from operator import attrgetter
import matplotlib.pyplot as plt
from _00_gene import *
import sys

def fitness(
	arr_close	:	np.ndarray	=	None,
	arr_low		:	np.ndarray	=	None,
	arr_returns	:	np.ndarray	=	None,
	arr_kratio	:	np.ndarray	=	None,
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
			pass
			#assert arr_close != None, \
			#	"Data is not specified, but close data was not provided to the fitness function."
			#if(method is martin_ratio):
			#	assert arr_low != None, \
			#		"Data is not specified, martin ratio is selected, but low data was not provided to the fitness function."
		case "form_519":
			
			#some_index = -1
			arr_close = data[:, 2]
			arr_low = data[:, 1]

	#boolean 2d array containing entry/or-not (0|1) for each gene
	#gene_presence = []

	gene_presence_local = []

	length = len(data)

	returns = []
	kelsch_ratio = []

	#test all samples in the set, accounting for
	#lag allowance and hold length
	for i in range(length):
		
		if(i < lag_allow | i > length-hold_for-1):
			#want to avoid usage of these values for safe analysis
			#gene_presence.append([0]*len(genes))
			gene_presence_local = np.array([0]*len(genes))
			
			returns.append(gene_presence_local*arr_returns[i])
			kelsch_ratio.append(gene_presence_local*arr_kratio[i])
		else:
		
			i_presence = []

			if(i%25000==0):
				sys.stdout.write(f"\r{i}")
				sys.stdout.flush()
		
			#check presence of each gene at each sample
			for g, gene in enumerate(genes):
				matches = True
				for p in gene._patterns:

					#some referrential point printouts if needed
					#print(f"p: {type(p)} at {g}")
					#print(f'v1,v2,l1,l2: {p._v1} {p._v2} {p._l1} {p._l2}')
					
					#if given pattern holds true
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
			#gene_presence.append(i_presence)

			gene_presence_local = np.array(i_presence)

			#since we have moved returns and kelsch ratio to an earlier step, append those values now
			returns.append(gene_presence_local*arr_returns[i])
			kelsch_ratio.append(gene_presence_local*arr_kratio[i])
			
	#gene_presence = np.array(gene_presence)
	returns = np.array(returns)
	kelsch_ratio = np.array(kelsch_ratio)
	
	
	'''#now going to take gene presence and define
	#returns and custom index values
	for i in range(length):
		
		#we do not need to account for illogical extreme values of i,
		#since that was already taken  into account when collecting gene presence above.
		#those values are all zero.
		#illogical extremes means hold_for at end of dataset, and lag allownace at beginning of dataset


		if(i < lag_allow | i > length-hold_for-1):
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
			

			if(ki_local == np.nan):
				print('nan found!')
			#if(ki_local != 0):
				#print(f"{ret_local} ++++++ {ki_local}")
			kelsch_ratio_local = (ret_local - (ki_local))
			#else:
			#	kelsch_ratio_local = 100 #force a maximum value

			kelsch_ratio.append(gene_presence[i]*(kelsch_ratio_local))

	returns = np.array(returns)
	kelsch_ratio = np.array(kelsch_ratio)'''
	
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
		local_avg_return = average_nonzero(returns[:, gi])
		local_avg_kelsch_ratio = average_nonzero(kelsch_ratio[:, gi])
		local_total_return = total_return(returns[:, gi])
		local_consistency = consistency(returns[:, gi])
		local_frequency = frequency(returns[:, gi])
		local_total_kelsch_ratio = total_return(kelsch_ratio[:, gi])
		local_martin_ratio = martin_ratio(returns[:, gi])
		local_mkr = martin_ratio(kelsch_ratio[:, gi])

		#if this is reached, this means the returns are coming in as the percent
		#difference for each trade IN LOG SPACE
		#and also that the kelsch ratios are coming in as the percent difference
		#minus standard deviation of drawdowns ALL IN LOG SPACE
		if(log_normalize):
			
			#bring them out of log space
			local_avg_return = np.exp(local_avg_return)-1
			local_avg_kelsch_ratio = (local_avg_kelsch_ratio)
			#these are now exact average % price differences (KR having some complexities)
			

		#update data within the gene for local storage for quick evaluation or recall
		gene.update(
			array_returns		=	returns[:, gi],
			#array_kelsch_ratio	=	kelsch_ratio[:, gi],
			avg_returns			=	local_avg_return,
			avg_kelsch_ratio	=	local_avg_kelsch_ratio,
			profit_factor		=	local_profit_factor,
			total_return		=	local_total_return,
			consistency			=	local_consistency,
			frequency			=	local_frequency,
			total_kelsch_ratio	=	local_total_kelsch_ratio,
			martin_ratio		=	local_martin_ratio,
			mkr					=	local_mkr
		)

	#returns updated genes
	return genes

def sort_population(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return','total_return','consistency']	=	'profit_factor'
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
		case 'total_return':
			metric = "last_total_return"
		case 'consistency':
			metric = "last_consistency"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({criteria})")
		
	sorted_pop = sorted(population, key=attrgetter(metric), reverse=True)

	#return population sorted by specified metric within each gene
	return sorted_pop

def show_best_gene_patterns(
	population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return']	=	'profit_factor',
	fss			:	list	=	None
):
	'''
	This function shows the basic data of the best gene in a list of genes (population)
	'''

	#variable to collect string
	output=""

	#sort population so we can grab the first guy
	s_p = sort_population(population,criteria)

	#use the pattern class built in function to show all patterns first
	output+=f"{s_p[0].show_patterns(fss)}"

	last_profit_factor = round(s_p[0]._last_profit_factor, 5)

	#collect more interpretable data for the return and KRatio
	#this is done by considering their values being percent differences, and
	#multiplying it by a very rough real price guesstimate, also assuming this is on SPY
	avg_return_ticks = round( (s_p[0]._lastavg_returns)*5000 , 2) #5k is super rough estimate on spy price
	avg_kratio_ticks = round( (s_p[0]._lastavg_kelsch_ratio)*5000 , 2) #5k is super rough estimate on spy price

	#then show basic metrics of the gene across last test
	output+=f"Profit Factor:	{str(last_profit_factor)}\n"
	output+=str(f"Average Return:	{str(round(s_p[0]._lastavg_returns,5))}	(~{avg_return_ticks} on /MES == ${round(avg_return_ticks*5, 2)})\n")
	output+=str(f"Average KRatio:	{s_p[0]._lastavg_kelsch_ratio}	(~{avg_kratio_ticks} on /MES == ${round(avg_kratio_ticks*5, 2)})\n")

	return output


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
	array	:	np.ndarray	=	None	
):
	#will be traditionally used for averaging returns or ratios
	filtered = array[array != 0]

	avg = (np.mean(filtered)) if filtered.size > 0 else 0

	return avg

def total_returns(
	array	:	np.ndarray	=	None
):
	return np.sum(array)



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

def consistency(
	returns	:	np.ndarray	=	None
):
	return 0

def frequency(
	returns	:	np.ndarray	=	None
):
	'''
	This doesnt happen to account for trades that were exactly zero but thats just how it will have to be
	'''
	return np.sum(returns == 0)


def simple_generational_stat_output(
	population	:	list	=	None,
	metric		:	str		=	None
):
	all_metrics = []

	#for each type of criteria added
	match(metric):
		#profit factor
		case 'profit_factor':
			metric = "last_profit_factor"
		#average return
		case 'average_return':
			metric = "lastavg_returns"
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "lastavg_kelsch_ratio"
		case 'total_return':
			metric = "last_total_return"
		case 'consistency':
			metric = "last_consistency"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({metric})")

	fetch_metric = attrgetter(metric)

	for gene in population:

		all_metrics.append(fetch_metric(gene))

	avg_metric = np.mean(all_metrics)
	top_metric = max(all_metrics)

	return avg_metric, top_metric

def load_custom_genes(
	gene_args	:	list
):	
	'''
	takes in a list of dict kwargs and creates sets of genes with them.
	'''
	

def show_returns(
	arr_returns	:	np.ndarray	=	None,
	arr_close	:	np.ndarray	=	None,
	gene_kwargs	:	any			=	None
):
	cum_pl = []
	total = 0

	base_pl = []
	base_tot= 0

	print(len(arr_returns))
	print(len(arr_close))

	for i, r in enumerate(arr_returns):
		total+=r
		cum_pl.append(total)
		base_tot=(arr_close[i]/arr_close[0])-1
		base_pl.append(base_tot)

	gene_info = show_best_gene_patterns(**gene_kwargs)

	plt.title(f"Percent Return of a Gene:\n{gene_info}")

	plt.plot(base_pl, color='black', label='Market Return')
	plt.plot(cum_pl,color='maroon', label='Strategy Return')

	plt.legend()
	plt.show()

	plt.plot(cum_pl,color='maroon')
	plt.show()

def filter_population(
	population	:	list	=	[],
	avg_return	:	float	=	-100,
	tot_return	:	float	=	0.0,
	profit_factor	:	float=	0,
	kelsch_ratio	:	float=	0,
	entry_frequency	:	float=	0.00
):
	'''
	This function takes a few different areas and filters the population based on such
	'''

	filtered_population = population

	pop_list = []

	for g, gene in enumerate(population):

		#check first for insufficient avg return
		if(gene._lastavg_returns < avg_return):
			#print(f"pop avg return {gene._lastavg_returns}")
			pop_list.append(g)
		elif(gene._last_total_return < tot_return):
			#print(f"pop tot return {gene._last_total_return}")
			pop_list.append(g)
		elif(gene._last_profit_factor < profit_factor):
			#print(f"pop prof fact {gene._last_profit_factor}")
			pop_list.append(g)
		elif(gene._lastavg_kelsch_ratio < kelsch_ratio):
			#print(f"pop kratio {gene._lastavg_kelsch_ratio}")
			pop_list.append(g)
		elif(gene._last_frequency < entry_frequency):
			#print(f"pop frequency {gene._last_frequency}")
			pop_list.append(g)

	pop_list = sorted(pop_list, reverse=True)

	for i in pop_list:
		filtered_population.pop(i)

	return filtered_population
