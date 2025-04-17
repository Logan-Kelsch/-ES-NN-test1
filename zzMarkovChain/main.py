'''
Logan Kelsch - 4/9/25
This file is created because I want to learn how to use markov chains.
I know nothing about this and want to teach myself. I seem to find that
These high level theoretical computational statements , formulas , and concepts
are easiest learned through application and program development.
'''

import numpy as np
import operator

def build_transition_matrix(
	trans_length	:	int		=	0,
	arr_states	:	np.ndarray	=	None,
	format		:	str			=	'beta_vhs',
	arr_filters	:	np.ndarray	=	None,
	filter_args	:	dict		=	{},
	col_filter	:	str			=	'None',
	dir_norm	:	str			=	'row'
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

	#simple operator mapping for ease of bringing in filtering information
	op_map = {
		'lt':operator.lt,
		'gt':operator.gt,
		'le':operator.le,
		'ge':operator.ge,
		'eq':operator.eq
	}

	#considering it is zero indexing, the length of states should equal the number of the highest number state.
	#if this is not true, the some theorized and computed state never showed up in the entire dataset.
	if(existing_states[-1] == len(existing_states)):
		print(f"States Never Seen: {np.setdiff1d(np.arange(existing_states[0], existing_states[-1]+1), existing_states)}")

	#using last value in array instead of length to ensure that if states never show up that they are at least shown
	n_states = existing_states[-1]+1

	#going to make a matrix where all transitions will be tossed into bins for each location of the matrix
	trans_tally_matrix = np.zeros((n_states, n_states), dtype=int)

	total_transitions = 0

	#loop through all samples, check to see if it is not filtered out, then evaluate where it goes in the transition matrix
	#NOTE not doing this right now, will remove filtered samples entirely from data existance -> #going to use state -1 (appended last state) as filtered out state 
	for i, state in enumerate(arr_states):

		#transition comparison will be done backwards, simple displacement sanity check
		if( i < trans_length ):
			continue

		#variable to check passing of filters
		keep = True
		
		#check to see if this sample is filtered out with any provided filters
		for f, (f_i, f_v) in enumerate(filter_args.items()):

			#now we are within a single features filters, need to iterate by pair 
			for p, pair in enumerate(f_v):

				#on filter f within filters
				#which is relating filter dataset column f_i
				#and comparing filter value with operator pair[0] over pair[1] value

				#if filter is NOTE TRUE NOTE TRUE, then remove
				if( op_map[pair[0]]( arr_filters[f_i] , pair[1] ) ):
					keep = False
					break
					
			#checked all pairs of given filter, is it being filtered out?
			if( keep == False ):
				break
				
		#if the loops were broken, go to next sample
		if( keep == False ):
			continue

		#if this area is reached, then it is not being filtered out 
		#and the sample needs to be accepted into the transition matrix
		trans_tally_matrix[ arr_states[ i-trans_length ], arr_states[ i ] ] += 1

		#tally for ease for normalization after tally
		total_transitions += 1

	match(col_filter):
		case 'None':
			pass
		case 'evens':
			trans_tally_matrix[:, ::2] = 0

	#normalizing i suppose is done by rows for markov
	match(dir_norm):
		case 'col':
			sums = trans_tally_matrix.sum(axis=0, keepdims=True)
			#wherever row_sum!=0 do the division, else put 0
			probability_matrix = np.where(
				sums != 0,
				np.transpose(np.transpose(trans_tally_matrix) / sums),
				0
			)
		case 'row':
			sums = trans_tally_matrix.sum(axis=1, keepdims=True)
			#wherever row_sum!=0 do the division, else put 0
			probability_matrix = np.where(
				sums != 0,
				trans_tally_matrix / sums,
				0
			)

	#wherever row_sum!=0 do the division, else put 0
	'''probability_matrix = np.where(
		sums != 0,
		trans_tally_matrix / sums,
		0
	)'''
	
	return probability_matrix


def fn_return_fss(
	mode	:	str	=	'beta_vhs'
):
	match(mode):
		case 'beta_vhs':
			return [[0, 1, 2, 3, 4],[5, 6],[7, 8, 9],[10,11]]
		case _:
			raise NotImplementedError(f"Your goofy mode '{mode}' is not goofy implemented for fss retrieval in Markov Chain folder.")
		
def build_state_array(
	data	:	np.ndarray,
	fss_mode:	str	=	'beta_vhs',
	mode	:	str =	'3to27'
):
	'''
	### info: ###
	this function will take all data instances within the time and day restrictions and create a state chain out of it
	### params: ###
	- data:
	-	-	the array of data of all features
	- time-rst:
	-	-	a tuple denoting the beginning and end time of allowable trade entries
	- day-rst:
	-	-	a tuple denoting the beginning and end day of allowable trading entries
	- fss-mode:
	-	-	a variable denoting the format of the dataset
	'''

	fss = fn_return_fss(mode=fss_mode)

	
	vhsi = fss[2]
	mai = fss[3]

	state_array = np.full(len(data), -1)

	match(mode):
		case '3to27':

			for sample in range(len(data)):

				tmp=-1

				if(data[sample,vhsi[0]]<20):
					tmp=0
				elif(data[sample,vhsi[0]]>80):
					tmp=2
				else:
					tmp=1


				if(data[sample,vhsi[1]]<20):
					pass
				elif(data[sample,vhsi[1]]>80):
					tmp+=6
				else:
					tmp+=3


				if(data[sample,vhsi[2]]<20):
					pass
				elif(data[sample,vhsi[2]]>80):
					tmp+=18
				else:
					tmp+=9

				state_array[sample] = tmp

		case '3ma15to12':

			for sample in range(len(data)):

				tmp=-1

				if(data[sample,vhsi[0]]>80 and data[sample,vhsi[1]]>50):
					tmp=1
				else:
					tmp=0


				if(data[sample,vhsi[2]]<20):
					pass
				elif(data[sample,vhsi[2]]>80):
					tmp+=4
				else:
					tmp+=2

				if(data[sample,mai[0]]>0):
					tmp+=6

				state_array[sample] = tmp

		case '3ma60to12':

			for sample in range(len(data)):

				tmp=-1

				if(data[sample,vhsi[0]]>80 and data[sample,vhsi[1]]>50):
					tmp=1
				else:
					tmp=0


				if(data[sample,vhsi[2]]<20):
					pass
				elif(data[sample,vhsi[2]]>80):
					tmp+=4
				else:
					tmp+=2

				if(data[sample,mai[1]]>0):
					tmp+=6

				state_array[sample] = tmp


	return state_array