import _00_gene as _0
import _01_initialization as _1
import _02_evaluation as _2
import _03_elitism as _3
import _04_parentselection as _4
import _05_reproduction as _5
import _06_mutation as _6

import numpy as np



def era(
	eon_num	:	int,
	era_num	:	int,
	new_population	:	np.ndarray,
	generations	:	int,
	dataset		:	np.ndarray,
	fss			:	list,
	criteria	:	str,
	log_normalize	:	bool,
	hold_for	:	int,
	lag_allowance	:	int,
	arr_returns	:	np.ndarray,
	arr_kratio	:	np.ndarray,
	elite_criteria	:	float|int,
	num_parents	:	int,
	rep_mode	:	str,
	with_array	:	bool,
	part_mproba	:	float,
	ptrn_mproba	:	float,
	use_strict_filter	:	bool,
	strict_filter_kwargs	:	dict
):
	
	stat_hold = (0, 0, 0)
	
	for generation in range(generations):

		population = new_population

		print(f"        EON {eon_num+1} ERA {era_num+1} GEN {generation+1} ({criteria}): ",end='')

		returns, kelsch_ratio = _2.fitness(
			arr_returns=arr_returns,
			arr_kratio=arr_kratio,
			data=dataset,
			genes= population,
			hold_for=hold_for,
			lag_allow=lag_allowance
		)
		unsorted_population = _2.associate(
			genes=population,
			returns=returns,
			with_array=with_array,
			kelsch_ratio=kelsch_ratio,
			log_normalize=log_normalize
		)

		if(use_strict_filter):
			unsorted_population = _2.filter_population(
				population=unsorted_population,
				**strict_filter_kwargs
			)

		population = _2.sort_population(
			population=unsorted_population,
			criteria=criteria
		)

		avg, top = _2.simple_generational_stat_output(population,criteria)

		if(avg+top == -2):
			print("Zero genes survived.")
			return []
		
		#make sure there is not a generation of no change
		if(((stat_hold[0] == avg) & (stat_hold[1] == top)) & (stat_hold[2] == len(population))):
			#this means nothing has changed since
			print(f"AVG {round(avg, 5)}, BEST {round(top, 5)}, FROM {len(population)} GENES.")
			return population
		else:
			stat_hold = (avg, top, len(population))
		
		print(f"AVG {round(avg, 5)}, BEST {round(top, 5)}, FROM {len(population)} GENES.")
		
		elites = _3.collect_elite(
			sorted_population=population,
			filter_criteria=elite_criteria
		)

		if(len(population) == 1):
			#no parents to collect, only elite
			return population

		parents = _4.collect_parents(
			sorted_population=population,
			criteria=criteria,
			num_parents=num_parents
		)

		family = _5.reproduce(
			parents=parents,
			mode=rep_mode
		)
		shuffled_family = _1.shuffle_population(
			population=family
		)
		mutated_family = _6.mutation_round(
			shuffled_population=shuffled_family,
			partial_mutation_prob=part_mproba,
			pattern_mutation_prob=ptrn_mproba,
			feat_idx_pool=fss	
		)
		new_population = _1.combine_populations(
			populations=[elites, mutated_family]
		)

	return population