#select two patterns to serve as parents
#they create two children patterns
#
#first sum all fitness values
#gen rand value between zero and pool sum
#create select sum
#loop throuhg pool, add pattern fitness
#first pattern where sel sum >=rand value is selected as parent
#
#if fitness is non-positive, cannot be selected as parent
#

#repeat process again

import random
from typing import Literal
from operator import attrgetter

def collect_parents(
	sorted_population	:	list	=	None,
	criteria	:	Literal['profit_factor','kelsch_ratio','average_return','total_return','consistency',\
							'frequency','total_kelsch_ratio','martin_ratio','mkr','r2','r2_kr']	=	'profit_factor',
	num_parents			:	int		=	2
):
	
	#create list variable for selected parents
	parents = []
    
	#variable used for selecting which metric to use in pool sum
	metric = ""
	metric_min = 0

	#for each type of criteria added
	match(criteria):
		#profit factor
		case 'profit_factor':
			metric = "profit_factor"
			metric_min = 1
		#average return
		case 'average_return':
			metric = "avg_returns"
			metric_min = 0
		#kelsch ratio
		case 'kelsch_ratio':
			metric = "avg_kelsch_ratio"
			metric_min = 0

		case 'total_return':
			metric = "total_return"
			metric_min = 0

		case 'consistency':
			metric = "consistency"
			metric_min = 0

		case 'frequency':
			metric = "frequency"
			metric_min = 0.001
		case "total_kelsch_ratio":
			metric = "total_kelsch_ratio"
			metric_min = 1

		case "martin_ratio":
			metric = "martin_ratio"
			metric_min = 0

		case "mkr":
			metric = "mkr"
			metric_min = 0

		case "r2":
			metric = "r2"
		#invalid entry, should be impossible anyways
		case _:
			raise ValueError(f"FATAL: Tried sorting population with invalid criteria ({criteria})")
		
	#make attribute getter
	fetch_metric = attrgetter(metric)

	#collect the total of fitness values
	pool_sum = 0

	#list of all metric values for ease of checking
	population_metrics = []

	#go thrugh all genes and add up metric values
	for gene in sorted_population:
		
		#fetch gene local metric
		fetched_metric = fetch_metric(gene)

		#ensure the gene is able to be selected
		if(fetched_metric>metric_min):
			pool_sum += fetched_metric
			population_metrics.append(fetched_metric)
		else:
			population_metrics.append(metric_min)

	#print(population_metrics)

	#for each parent selected, do the following
	for p in range(num_parents):

		#generate a random variable between the minimum metric and the pool sum
		survival_threshold = random.uniform(metric_min, pool_sum)

		#sum up metrics until it is greater than the threshold, save as a parent
		selection_sum = 0

		#go through all genes searching for first sample above survival threshold
		for gene in sorted_population:

			#fetch the relevant metric
			fetched_metric = fetch_metric(gene)

			#ensure the gene is able to be selected
			if(fetched_metric>metric_min):
				selection_sum += fetched_metric

			#check if the survival threshold was surpassed
			if(selection_sum >= survival_threshold):
				
				#appended gene that survived first to list
				parents.append(gene)
				#and move into next parent selection
				break

	#return list of selected parents
	return parents
