import _0_gene as _0
import random

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