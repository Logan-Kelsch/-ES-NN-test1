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