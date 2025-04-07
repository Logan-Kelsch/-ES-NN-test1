import _00_gene as _0
import random
import numpy as np
from math import sqrt
from typing import Literal

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
	direction:	bool		=	1,
	arr_close:	np.ndarray	=	None,
	arr_ext	:	np.ndarray	=	None,
	exit_mode:	any			=	0,
	exit_type:	Literal['area','line']	=	'area',
	exit_cond:	any			=	None,
	dataset:	np.ndarray	=	None,
	lag_allow:	int			=	0,
	log_normalize:	bool	=	True
):
	'''
	This function is going to take a given dataset and return the essential parallel data<br>
	such as:
	- kelsch ratio of given holdfor length
	- returns of given holdfor length

	NOTE EXPANDED FUNCTIONALITY EXPLANATION NOTE
	This function will now support the use of custom single dimension exit conditioning.
	This will also still have hold_for functionality, and as default.
	Takes in an dataset and fss list.
	creates 1 DIMENSIONAL exit signal from a single feature of the dataset.
	for instance, using custom feature based exit conditioning:
	>>> grab dataset
	>>> grab and stash (hawkes stoch at 60 mins and .08 kappa) as array
	>>> bring in some exit conditioning, such as (stashed feature is less than .2)
	>>> before the loop is created, go through entire dataset and create an array
	>>> This array will be used for collecting exit time displacement information, or holding an illegal flag
	>>> this array holds state info of the following:
	>>> time-till-exit, or forbidden of entry. These are denoted as
	>>> n (candles), 0 (ln of (price1/price1) will return PL of zero, simulating no value in entry)
	Some candles are considered forbidden (of trade) if exit_type is area, and exit condition is satisfied. 
	'''

	assert exit_cond != None, 'Exit condition was never specified. Check documentation of collecting parallel metrics.'

	#list variables for holding metrics
	returns = []
	kelsch_ratio = []

	length = len(arr_close)

	exit_disp = np.zeros(length, dtype=np.float32)

	#this loop is for collecting an array containing how long a trade would be held for at each instance of the backtesting data
	#forbidden trade areas hold zeros to denote zero holding time
	#hold_for (initial functionality of this file) sets a constant value for all instances of the dataset (ex; 15 mins holding time)
	for i in range(length):

		#check which trading mode is being used
		match(exit_mode):

			#classical use of this file, always holds trade for set amount of time
			case 0|'hold_for':
				exit_disp = exit_cond

			#a custom exit statement is being created
			case 1|'custom':
				
				#exit_cond should come in as a tuple containing
				#the index of desired conditional variable in the dataset
				#the comparator, denoted as a string of either 'lt','gt','le','ge'
				#the comarative variable
				#an example of this could be a bollinger band brought in, where exit is -1 (stdev from ma)

				'''NOTE ENSURE THAT VARIABLES ARE CALLED LEGALLY, consider trimming conditional statements if needed'''
				'''resume coding here -	-	-	-	-	-	-	-	- NOTE NOTE NOTE NOTE'''

				'''
				Current psuedo code:
				
				grab the feature to use
				isolate that into a variable for minimizing calcuations
				check each sample instance for exit or forbidden state presence
				then go back and collect time until for each
				consider making holding cache for each instance.
				'''
				pass

	for i in range(length):

		hold_for = exit_disp[i]

		if((i < lag_allow) | (i > length-hold_for-1)):
			#want to avoid usage of these values for safe analysis
			returns.append(0)
			kelsch_ratio.append(0)
		else:
		
			#calculate returns
			if(log_normalize):
				if(direction==1):
					returns_local = np.log(arr_close[i+hold_for]/arr_close[i])
				else:
					returns_local = np.log(arr_close[i]/arr_close[i+hold_for])
			else:
				if(direction==1):
					returns_local = (arr_close[i+hold_for] - arr_close[i])
				else:
					returns_local = (arr_close[i] - arr_close[i+hold_for])

			returns.append(returns_local)
			
			#calculate index values here if desired
			ki_local = 0
			entry_price = arr_close[i]
			for c in range(1,hold_for+1):
				if(log_normalize):
					if(direction==1):
						ki = entry_price/arr_ext[i+c] #>=1 means is below entry
					else:
						ki = arr_ext[i+c]/entry_price

					ki_local+=(np.log(max(ki, 1))**2)
				else:
					if(direction==1):
						ki_local+=((max(entry_price - arr_ext[i+c] , 0)) ** 2)
					else:
						ki_local+=((max(arr_ext[i+c] - entry_price, 0)) ** 2)
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

			if(kelsch_ratio_local > 10):
				kelsch_ratio_local = 10

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

def make_population_batches(
	population	:	list,
	batch_size	:	int	=	100
):
	return [population[i:i + batch_size] for i in range(0, len(population), batch_size)]