import _00_gene as _0
import random
import numpy as np
from math import sqrt

def generate_initial_population(
	sample_size	:	int	=	None,
	pattern_size:	int	=	None,
	feat_idx_pool:	list=	[],
	lag_allowance:	int	=	None,
	skew_lag_prob:	bool=	False
):
	#initial generation of genes list
	genes = []

	#create lag list/range based on skew boolean
	if(skew_lag_prob):
		raise NotImplementedError('FATAL: skewing lag probability has not been implemented yet.')
		lags = None
	else:
		lags = list(range(lag_allowance+1))

	#creation of each gene and patterns for each
	for gene in range(sample_size):

		#create patterns local to each gene
		local_patterns = []

		#each pattern creation is here, appended to local_patterns
		for pattern in range(pattern_size):

			local_patterns.append(
				_0.Pattern(

				#pull in lag values from earlier declaration
				acceptable_lags = lags,
				
				#choose a feature subset to save logically-related feature values
				acceptable_vals = random.choice(feat_idx_pool)
				
				)
			)

		#add each gene created to the list of genes
		genes.append(_0.Gene(
			patterns = local_patterns
		))

	#returns the list of created genes
	return genes

def collect_parallel_metrics(
	arr_close:	np.ndarray	=	None,
	arr_low	:	np.ndarray	=	None,
	hold_for:	int			=	0,
	lag_allow:	int			=	0,
	log_normalize:	bool	=	True
):
	'''
	This function is going to take a given dataset and return the essential parallel data<br>
	such as:
	- kelsch ratio of given holdfor length
	- returns of given holdfor length
	'''
	
	#list variables for holding metrics
	returns = []
	kelsch_ratio = []

	length = len(arr_close)

	for i in range(length):
		if(i < lag_allow | i > length-hold_for-1):
			#want to avoid usage of these values for safe analysis
			returns.append(0)
			kelsch_ratio.append(0)
		else:
		
			#calculate returns
			if(log_normalize):
				returns_local = np.log(arr_close[i+hold_for]/arr_close[i])
			else:
				returns_local = (arr_close[i+hold_for] - arr_close[i])
			returns.append(returns_local)
			
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
			
			#this comes up when there is zero closes below the entry close
			if(ki_local == 0):
				kelsch_ratio_local = 10
			#take difference to show differential between std drawdown and profit
			else:
				kelsch_ratio_local = ((returns_local) / (ki_local))

			kelsch_ratio.append((kelsch_ratio_local))

	#metrics are built, return parallel metrics
	return returns, kelsch_ratio



def combine_populations(
	populations	:	list	=	None
):
	'''
	This function will take a list OF POPULATIONS and turn them into a single population list
	'''

	#create variable to combine populations
	combined_populations = []

	#run nested loop to collect all genes from all populations
	for population in populations:
		for gene in population:

			#add gene to combined pool
			combined_populations.append(gene)

	#returns single list of all populations members
	return combined_populations

def shuffle_population(
	population	:	list	=	None
):
	random.shuffle(population)
	return population