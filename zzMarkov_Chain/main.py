'''
Logan Kelsch - 4/9/25
This file is created because I want to learn how to use markov chains.
I know nothing about this and want to teach myself. I seem to find that
These high level theoretical computational statements , formulas , and concepts
are easiest learned through application and program development.
'''

import numpy as np

def build_transition_matrix(
	arr_states	:	np.ndarray	=	None,
	arr_filters	:	np.ndarray	=	None,
	filter_args	:	dict		=	{}
):
	'''
	 ### info: ###
	 This function takes in a given array of states over time and returns the transition matrix
	 ### params: ###
	 -	arr-states:
	 -	-	an array of length dataset containing state information
	 -	arr-filters:
	 -	-	an array of any features that will be used for sample filtering (ex: tod)
	 -	filter-args:
	 -	-	a dict containing interpretation instructions for arr-filters. formed as:
	 -	-	key - (int) arr-filters feature index
	 -	-	value-(tuple) a tuple of tuples (pairs) containing data comparative instructions.
	 -	-	ex: (('gt', 570), ('lt', 900)) meaning keep minutes between 9:30am-3:00pm
 	'''

	#first off, collect unique values from states array
	existing_states = np.unique(arr_states)

	#considering it is zero indexing, the length of states should equal the number of the highest number state.
	#if this is not true, the some theorized and computed state never showed up in the entire dataset.
	if(existing_states[-1] == len(exitsting_states)):
		print(f"States Never Seen: {np.setdiff1d(np.arange(existing_states[0], existing_states[-1]+1), existing_states)}")

	#using last value in array instead of length to ensure that if states never show up that they are at least shown
	n_states = existing_states[-1]

	#going to make a matrix where all transitions will be tossed into bins for each location of the matrix
	trans_total_matrix = np.zeros((existing_states), dtype=int)

	return


									
