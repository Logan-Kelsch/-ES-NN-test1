import _00_gene as _0
import random
import numpy as np
from math import sqrt
from typing import Literal
import operator

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
	exit_cond:	any			=	None,
	clip_prof:	float		=	0,
	clip_loss:	float		=	0,
	dataset:	np.ndarray	=	None,
	lag_allow:	int			=	0,
	log_normalize:	bool	=	True
):
	'''
	This function is going to take a given dataset and return the essential parallel data<br>
	such as:
	- kelsch ratio of given holdfor length
	- returns of given holdfor length
	### params: ###
	- 
	- 
	- 
	- 
	- clip_prof:
	-	-	take profit, in percent
	- clip_loss:
	-	-	stop loss, in percent

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

	exit_disp = np.zeros(length, dtype=int)

	#this loop is for collecting an array containing how long a trade would be held for at each instance of the backtesting data
	#forbidden trade areas hold zeros to denote zero holding time
	#hold_for (initial functionality of this file) sets a constant value for all instances of the dataset (ex; 15 mins holding time)
	#for i in range(length):

	#check which trading mode is being used
	match(exit_cond):
		#classical use of this file, always holds trade for set amount of time
		case int()|float():
			exit_disp = np.full(length, fill_value=exit_cond, dtype=int)

		#a custom exit statement is being created
		case tuple():

			exit_disp = np.full(length, fill_value=-1, dtype=int)

			op_map = {
				'lt'	:	operator.lt,
				'gt'	:	operator.gt,
				'le'	:	operator.le,
				'ge'	:	operator.ge,
				'eq'	:	operator.eq
			}
			
			#exit_cond should come in as a tuple containing
			#the index of desired conditional variable in the dataset
			#the comparator, denoted as a string of either 'lt','gt','le','ge'
			#the comarative variable
			#an example of this could be a bollinger band brought in, where exit is -1 (stdev from ma)

			#grab the feature to use and isolate for minimal calculation
			exit_feat = dataset[:,exit_cond[0]]

			#grab the comparing operator
			exit_op = op_map[exit_cond[1]]

			#grab the comparing value
			exit_comp = exit_cond[2]

			next_satisfied = -1

			#splitting exit type pre-sample-loop to minimize time-complexity
			match(exit_cond[3]):

				#this means if condition is satisfied, is forbidden area
				case 'area':

					#honestly used chatgpt for this idea, absolutely smartmode idea it came up with!!
					for i in reversed(range(len(exit_feat))):

						#check to see if condition is satisfied
						if(exit_op(exit_feat[i], exit_comp)):

							next_satisfied = i

						#end of week, do not stay in trade
						if(dataset[i,5]==1019 and dataset[i,6] == 5):

							next_satisfied = i
						
						if(next_satisfied != -1):

							exit_disp[i] = next_satisfied - i

				#this means only forbidden area is the moment that condition is satisfied, 
				#if subsequently followed by condition satisfaction, it is neglected and is not forbidden 
				case 'line':

					#honestly used chatgpt for this idea, absolutely smartmode idea it came up with!!
					for i in reversed(range(len(exit_feat))):

						#check to see if condition is satisfied, and also that it was not satisfied the previous minute
						#this is an identification of a cross in the conditional truth space
						if(exit_op(exit_feat[i], exit_comp) and not exit_op(exit_feat[(i-1)%len(exit_feat)], exit_comp)):

							next_satisfied = i

						#end of week, do not stay in trade
						if(dataset[i,5]==1019 and dataset[i,6] == 5):

							next_satisfied = i
						
						if(next_satisfied != -1):

							exit_disp[i] = next_satisfied - i

	#entire exit_disp array is completed, all match cases are exited
	#ensure any forbidden area flags (-1) are converted to zero
	#zero as an exit distance measurement is desireable as it will be thrown into calculation where:
	#we are taking natural log of exit price over entry price. if zero exit displacement then it is
	#some price n over some price n => equals 1 => ln(1) = 0, making zero return undesirable/unnoticable
	#now negating all significance of any trade here, making model gene pool ignore these areas! 
	#perfect woop woop woop woop woopen gangnem style
	exit_disp[exit_disp == -1] = 0 		

	#identify pre-calculated log values of clips to minimize computation
	log_clip_prof = np.log(1+clip_prof)
	log_clip_loss = -np.log(1+clip_loss)

	#now we are going to loop through the array with the provided exit displacement variable array
	#and calculate returns from all bars under these circumstances
	for i in range(length):

		hold_for = exit_disp[i]

		if((i < lag_allow) | (i > length-hold_for-1)):
			#want to avoid usage of these values for safe analysis
			returns.append(0)
			kelsch_ratio.append(0)
		else:
			
			#forbidden case, can avoid calculation
			if(hold_for == 0):
				returns_local = 0

			#calculate returns
			elif(log_normalize):
				if(direction==1):
					returns_local = np.log(arr_close[i+hold_for]/arr_close[i])
				else:
					returns_local = np.log(arr_close[i]/arr_close[i+hold_for])

				if((clip_loss!=0.0) or (clip_prof!=0.0)):
					if(returns_local > log_clip_prof):
						returns_local = log_clip_prof
					if(returns_local < log_clip_loss):
						returns_local = log_clip_loss

			else:
				if(direction==1):
					returns_local = (arr_close[i+hold_for] - arr_close[i])
				else:
					returns_local = (arr_close[i] - arr_close[i+hold_for])

				if((clip_loss!=0.0) or (clip_prof!=0.0)):
					if(returns_local > clip_prof):
						returns_local = clip_prof
					if(returns_local < clip_loss):
						returns_local = clip_loss


			returns.append(returns_local)
			
			if(hold_for == 0):
				kelsch_ratio_local = 0
			else:
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