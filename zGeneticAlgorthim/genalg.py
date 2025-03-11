'''
Logan Kelsch	-	Initialized 3/11/2025
This file will be for the actual shaping and execution of the genetic algorithm.
'''

import _1_initialization as initialization
import _2_evaluation as sample_evaluation

import numpy as np


def genetic_algorithm(
	some_params
):
	'''
 	func info here
  	'''

	#form initial population
	population = initialization.generate_initial_population(
		sample_size, 
		pattern_allowance, 
		lag_allowance
	)
	
	#this population variable will be some unordered list of Genes

	#make a variable that runs parallel to population to evaluate fitness
	gene_fitness = [None] * len(population)
	
	for g_index, gene in enumerate(population):
		pass
		#test some fitness here
		#possibly enumerate through data instead of genes? whihc way to nest?
