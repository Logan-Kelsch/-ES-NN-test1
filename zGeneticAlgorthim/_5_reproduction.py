
from typing import Literal
from itertools import combinations
import random
import _0_gene as _0

def reproduce(
	parents	:	list	=	None,
    mode	:	Literal['linear','exponential']	=	'exponential'
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

	#gather number of patterns in each gene
	num_patterns = len(parents[0]._patterns)

	#gather number of parents brought into function
	num_parents = len(parents)

	#create children differently based off of selected mode
	match(mode):

		#linear is selected when only n children are derived from n parents with m patterns
		case 'linear':
			
			#create variable to make a random split somewhere in the patterns
			split = random.randint(1,num_patterns)

			#create shuffled list of parent indices for random linear size pairing
			parent_pairing_idx = random.shuffle(list(range(num_parents)))

			#append all parents to family list
			for p in parents:
				family.append(
					_0.Gene(
						patterns=parents[p]._patterns
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
						if(pattern_pool[i].equals(pattern_pool[i])):

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

			#generate all possibly combinations of these patterns. thanks to
			#a wonderful library I don't have to code this myself.
			#NOTE WITHOUT REPLACEMENT !! END#NOTE
			all_pattern_combos = combinations(pattern_pool, num_patterns)

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