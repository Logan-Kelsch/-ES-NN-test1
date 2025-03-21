#elite samples are exempt from mutation


#replace entire pattern with new pattern

import random

def mutation_round(
	shuffled_population		:	list	=	None,
	partial_mutation_prob	:	float	=	0.025,
	pattern_mutation_prob	:	float	=	0.025,
	feat_idx_pool			:	list	=	None
):

	#will need length of patterns
	len_patterns = len(shuffled_population[0]._patterns)

	#will need numbers of genes
	len_genes = len(shuffled_population)

	#create tally on number of mutations that will be made
	partial_mutations = multiple_chances(partial_mutation_prob,len_genes)
	pattern_mutations = multiple_chances(pattern_mutation_prob,len_genes)

	#perform all partial mutations on random samples
	for partial in range(partial_mutations):

		#go to random gene, go to random pattern, call switch on random pattern characteristic
		shuffled_population[random.choice(range(len_genes))]\
			._patterns[random.choice(range(len_patterns))].switch(type='mutation')
		
	#perform all partial mutations on random samples
	for pattern in range(pattern_mutations):

		#go to random gene, go to random pattern, call random to fully randomize pattern
		shuffled_population[random.choice(range(len_genes))]\
			._patterns[random.choice(range(len_patterns))].random(mutation=True,fss=feat_idx_pool)

	#return mutated population
	return shuffled_population		


#this function generates boolean, true percent% of the time
def chance(
	percent:float=0.5
):

	#if percent comes in as integer
	if(percent>1):
		percent/=100

	#if chance falls within percent window
	if(random.random()<percent):
		return True
		
	else:
		return False
	
def multiple_chances(
	percent	:	float	=	0.5,
	times	:	int		=	2
):
	'''
	really taking a break on the math for this one hahahaha.
	This can be easily mathed out but i do not feel like double checking
	trying to knock this out
	'''

	#tally number of instances is true
	instances = 0

	#check the chance by single instance
	for i in range(times):
		instances+=int(chance(percent))

	return instances