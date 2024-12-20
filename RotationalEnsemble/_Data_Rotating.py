'''
-   Data Rotating      - - Should end with an array of 'rotations' of data for n models from the train split
-   -   -   This should have a few options, pick random from all, option to keep some features in all models, ... 
-   -   -   -   ... random pick but category specific, overlapping or model unique features
'''


'''
moving into this finally.

NOTE 1
things to plan on building:
	
-	there should be a "ZERO" case for this function, where
given a certain (default) function parameter entry, there 
is no effect on the data at all except for the transformation
across the entire feature space through a rotation matrix
(which we can assume will be with the eigenvector matrix)


NOTE 2 

two variables 

This should end with n 'rotations' of the feature space [1-len(features)]
which will be an array of n variables, all being a set of training data for models.
'''

from _Utility import *
from typing import Literal, Union
import traceback
import random
import numpy as np

def data_partition(
	X
	,feat_sbst	:	list		=	[]
	,num_parts	:	int			=	5
	,part_type	:	Literal['full_set','by_subset']	=	'by_subset'
	,full_excl	:	bool		=	False
	,univ_incl	:	list		=	[]
	,part_sbst	:	float		=	0.5
)	->	list:
	'''
		This function is used to split the data for future rotation.
		It will return a list of (training) data partitions.
  Params:
- X:
-	_Training set of data.
- feat-sbst: 
-	_List of feature-subsets (as dict of df-index:feature-name)
- num-parts:
-	_Number of partitions made.
- part-type:
-	_Allows user to partition randomly through entire featurespace, or randomly and equally through each feature-subset-space.
- full-excl: 
-	_Option to disable ANY feature overlap across partitions.
- univ-incl: 
-	_List of feature names that will be included in all partitions. 
- part_sbst:
-	_The fraction (%) of each feature-subset that is included in each partition.
	'''
	if(full_excl):
		print('FATAL: FULL EXCLUSION OF FEATURES HAS NOT BEEN IMPLEMENTED.')
		traceback.print_exc()
		raise

	#pull list of featuresets into one dict
	all_feats = {k: v for d in feat_sbst for k, v in d.items()}

	#total number of features in each partition
	num_feats = int(np.floor(len(all_feats) * part_sbst))
 
	#an array (of training sets) of different selections of the featurespace
	X_partition = []

	#split function cases first by partition type:
	match(part_type):
     
		#features will be picked out of one basket
		case 'full_set':
			
			#for the number of requested partitions
			for partition in num_parts:
       
				#include all univeral inclusion features to each partitions features
				feature_indices = [k for k, v in all_feats.items() if v in univ_incl]
    
				#grab n TOTAL features (n minus univ OTHER features) from featureset
				feature_indices+= pick_n_from_list(num_feats - len(feature_indices), \
						[i for i in list(all_feats.keys()) if i not in feature_indices])
    
				#create new partitions for each
				X_partition.append(X[:, feature_indices])
				
		#features will be picked with consideration of feature-set
		case 'by_subset':
		
			#for each partition of requested total
			for partition in num_parts:
       
				#include all univeral inclusion features to each partitions features
				feature_indices = [k for k, v in all_feats.items() if v in univ_incl]
       
				#for each feature set of the feature_subsets variable
				for featset in feat_sbst:
        
					#ceil to lean towards inclusivity of features in small featuresets
					num_feats_from_here = int(np.ceil(len(featset) * part_sbst))
     
					#ensuring inclusive ceil function does not push out of bounds
					num_feats_from_here = len(featset) if \
         			(num_feats_from_here>len(featset)) else num_feats_from_here
     
					#add random picks from each subset
					feature_indices+= pick_n_from_list(num_feats_from_here, \
         												list(featset.keys()))
     
				#quickly remove all duplicates, possible here with complex
				#exclusion of subset (universal inclusion) for each featureset,
				#so I did not implement that, should not impact much considering
				#wide functionality range of this model.
				feature_indices = list(set(feature_indices))
    
				#create new partitions for each
				X_partition.append(X[:, feature_indices])

		#illegal case, neither option (of 2) entered
		case _:
			print("Illegal input for parameter 'part_type'.")
			print("Must be Literal['full_set','by_subset']")
			traceback.print_exc()
			raise

	#an array of training sets of different selections of the featurespace
	return X_partition


def data_rotation(
	rotn_type	:	Literal['PCA','Other']			=	'PCA'
	,filter		:	bool							=	False
	,fltr_type	:	Literal['Retention','Count']	=	'Retention'
	,fltr_param	:	Union[float, int]				=	1.0
):
	'''
		This function takes in a set of data and applies a rotation along the
		feature space.
  
  Params:
- rotn-type:
-	_Type of rotation (transformation) applied to the feature-space.
-	_Currently only using PCA, can implement more later.
- filter:
-	_Decide whether or not to decrease dimentionality of feature-space (post-rotation)
- fltr-type:
-	_Method of decreasing dimetionality. (Data retention (%), Dimension Count)
- fltr-param:
-	_Parameter for filtering. (Percent of data retained, Number of PC retained)
	'''
	
	return #a new dataset that is transformed by eigenvector matrix multiplication
														#through PCA decomposition