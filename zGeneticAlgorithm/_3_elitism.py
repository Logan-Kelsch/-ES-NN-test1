#select single best or percent of best and return as new list of population

#this way best always remains
#should be used sparingly
#dont forget other half of this is parent seleciton

def collect_elite(
	sorted_population:	list		=	None,
	filter_criteria	:	int|float	=	1
):
	'''
	### info: ###
	This function takes a sorted population and returns a list of the most elite genes.
	### params: ###
	- sorted-population:
	-	-	A population (list) of genes pre-sorted in terms of pre-specified metrics.
	- filter-criteria:
	-	-	The amount of genes to keep, interpretable as a percent (decimal) or exact count (integer)
	### returns: ###
	A list variable of the most elite genes (sub-population, sorted)
	'''
	

	#logical assertions
	assert (sorted_population != None), "Cannot have None population for selecting elite genes."
	assert (filter_criteria > 0), "Cannot have Zero/Negative Filter Criteria in elitism step."

	#will create a mode based off of what the filter criteria contains
	if(filter_criteria < 1):
		mode = 'percent'
	else:
		mode = 'count'

	#size variable
	pop_size = len(sorted_population)
	#how many to take
	take = 1
	#variable to collect elite genes
	elite_genes = []

	#approach each elite filtering based off of filter type brought in
	match(mode):

		#if the number comes in as a percent, interpret as such
		case 'percent':

			#push to lower end of percent at same time as casting to int
			take = int(filter_criteria * pop_size)
			#ensure take forms a valid number
			take = 1 if (take < 1) else (take)

		#if the number comes in as an integer, interpret as exact elite count
		case 'count':

			take = filter_criteria if (filter_criteria > 1) else (1)

	#append each most elite members to the elite gene population
	for i in range(take):
		elite_genes.append(sorted_population[i])

	#return the collected elite genes in list form
	return elite_genes

