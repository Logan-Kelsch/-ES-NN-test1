
from typing import Literal
from itertools import combinations
from operator import attrgetter
import numpy as np
import random
import copy
import _00_gene as _0

def reproduce(
	parents	:	list	=	None,
    mode	:	Literal['linear','exponential']	=	'exponential',
	custom_num_patterns	:	int	=	-1
):
    
	#parent pattern-sets p are brought into this function
    #each pattern-set is of size n
    #we will generate c child pattern-sets
    #each child's pattern-set will be a unique combination of
    #patterns of size m across patterns in p
    #where c's pattern-set = length m = (p_ij, p_uv, ...)
    
	#child list variable to collect produced children
	#this will now include parents too , renamed to family
    #list variable of gene class variables
	family = []

	#if a custom number of patterns is not used, this value comes in as -1
	#the number of patterns is then collected from parents data
	#ex: parents have n patterns therefore children have exactly n patterns.
	if(custom_num_patterns == -1):
		#gather number of patterns in each gene
		num_patterns = len(parents[0]._patterns)
	else:
		num_patterns = custom_num_patterns

	#gather number of parents brought into function
	num_parents = len(parents)

	#create children differently based off of selected mode
	match(mode):

		#linear is selected when only n children are derived from n parents with m patterns
		case 'linear':
			
			#create variable to make a random split somewhere in the patterns
			split = random.randint(1,num_patterns)

			#create shuffled list of parent indices for random linear size pairing
			
			parent_pairing_idx = list(range(num_parents))
			random.shuffle(parent_pairing_idx)

			#append all parents to family list
			for p in parents:
				family.append(
					_0.Gene(
						patterns=p._patterns
					)
				)

			#for each child created
			for i in range(num_parents):
				
				#create a variable to hold all new patterns
				new_child_patterns = []

				#grab each parent 1 split pattern from parent 1
				#parent 1 is selected at random from the static random parent index list
				for p1_patterns in range(0, split):
					new_child_patterns.append(parents[parent_pairing_idx[i]]._patterns[p1_patterns])

				#grab each parent 2 split pattern from parent 2
				#parent 2 is selected and unique by taking parent 1 random index and moving over 1, with looping precaution
				for p2_patterns in range(split, num_patterns):
					new_child_patterns.append(parents[parent_pairing_idx[(i+1)%num_parents]]._patterns[p2_patterns])

				#now create a child with these patterns!
				family.append(
					_0.Gene(
						patterns=new_child_patterns
					)
				)

		#exponential is selected when all pattern combinations are derived from n parents with m patterns
		case 'exponential':

			#variable to collect all patterns from all parents
			pattern_pool = []

			#add all patterns to the pool
			for parent in parents:
				for pattern in parent._patterns:
					#add each pattern from each parent
					pattern_pool.append(pattern)

			#check for pattern duplication first
			#make full check boolean variabel
			fully_unique = False

			#keep iterating through until all duplicates are removed
			while(fully_unique == False):

				#reset unique boolean checker
				fully_unique = True
				
				#use bubbling check, double iterating forward
				for i in range(len(pattern_pool)):
					for j in range(i+1, len(pattern_pool)):

						#this comparator lands at every pair, compairs patterns
						if(pattern_pool[i].equals(pattern_pool[j])):

							#if they are identical, define bool to allow for double escape
							fully_unique = False
							#remove the identical pattern
							pattern_pool.pop(j)
							#break from first loop
							break

					#check if bool was redefined, double escape if so
					if(fully_unique == False):
						break

			#now pattern_pool is a fully unique collection
			#shuffle all attributes
			random.shuffle(pattern_pool)

			#print(f'pattern pool len = {len(pattern_pool)}')

			#generate all possibly combinations of these patterns. thanks to
			#a wonderful library I don't have to code this myself.
			#NOTE WITHOUT REPLACEMENT !! END#NOTE
			all_pattern_combos = list(combinations(pattern_pool, num_patterns))
			#print(f'combo len = {len(all_pattern_combos)}')

			#now create a fresh gene of all combinations and add them to the family list
			for pattern in all_pattern_combos:
				family.append(
					_0.Gene(
						#pattern should be in tuple form, need to cast to list variable
						patterns= list(pattern)
					)
				)
    
	#return formed children
	return family


def evolutionary_branch(
	gene	:	any,
	branch_size	:	int,
	proba_num_mutations	:	list	=	[0.20, 0.45, 0.25, 0.10],
	proba_var_mutations	:	list	=	[0.13, 0.35, 0.04, 0.13, 0.35],
	include_original	:	bool	=	True
):
	'''
	### info: ###
	#### This function: 
	-	takes a SINGLE gene,
	-	creates an evolutionary branch..
	-	of num_mutations number of genes
	### params: ###
	- gene:
	-	-	the parent gene coming in
	- branch-size:
	-	-	the number of mutated members desired
	- proba-num-mutations:
	-	-	probabilities of number of mutations per pattern, starting from 1
	- proba-var-mutations:
	-	-	probabilities for each variable being mutated in a given instance
	- include-original:
	-	-	boolean for in the original gene is in the returned population
	### returns: ###
	a population list of mutated genes (family branch)
	'''

	#get the number of patterns that are in each gene
	num_patterns = len(gene._patterns)

	#variable to hold the family branch of genes (population)
	family_branch = []

	#check if user wants the original in the family branch
	if(include_original):
		family_branch.append(gene)

	#for ease of reference, actually I think there is an easier way, too late LOL
	attr_map = {
		0	:	"v1",
		1	:	"l1",
		2	:	"op",
		3	:	"v2",
		4	:	"l2"
	}

	#for each new gene mutation
	for i in range(branch_size):

		#make a local duplicate of the gene coming in
		new = copy.deepcopy(gene)

		#for each pattern of the parent gene
		for pattern in range(num_patterns):

			#decide how many mutations are made per pattern
			num_mutations = np.random.choice(list(range(1,5)), p=proba_num_mutations)

			#create a list of variables that will be mutated
			spec_mutations = np.random.choice(list(range(5)), size=num_mutations, replace=False, p=proba_var_mutations).tolist()

			#for each variable index to mutate
			for mutate in spec_mutations:

				#call switch for new variable value on the chosen pattern variable to mutate
				new._patterns[pattern].switch(attr_map[mutate])

		#by here, all patterns have been mutated to desired extent, local gene i is ready.
		family_branch.append(new)

	#by here, all mutated family members have been added to the family branch
	return family_branch


def additive_branch(
	gene	:	any,
	branch_size	:	int,
	fss			:	list
):
	'''
	### info: ###
	This function takes a given gene and turns it into a population of the parent gene with random patterns appended
	'''

	family_branch = []

	for i in range(branch_size):


		new = copy.deepcopy(gene)

		ptrn = _0.Pattern(
			acceptable_lags=new._patterns[0]._acceptable_lags,
			acceptable_vals=random.choice(fss))
		
		
		#ptrn.random(mutation=True,fss=fss)

		#print(ptrn.show(fss=fss))

		new._patterns.append(ptrn)


		family_branch.append(new)

	print(f"Expanded gene to {len(family_branch)} genes.")

	return family_branch

def expansive_recomposition(
	population	:	list,
	num_patterns	:	int
):
	'''
	### info: ###
	this function takes a population and generates a new one consisting of all possible combinations from the initially dissolved population pattern pool.
	'''

	init_size = len(population)

	pop_recomp = reproduce(
		parents=population,
		mode='exponential',
		custom_num_patterns=num_patterns
	)

	finl_size = len(pop_recomp)

	print(f"Expanded population: {init_size} -> {finl_size} at length {num_patterns}.")

	return pop_recomp