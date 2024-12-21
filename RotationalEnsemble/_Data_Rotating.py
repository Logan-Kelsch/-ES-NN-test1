'''
-   Data Rotating      - - Should end with an array of 'rotations' of data for n models from the train split
-   -   -   This should have a few options, pick random from all, option to keep some features in all models, ... 
-   -   -   -   ... random pick but category specific, overlapping or model unique features
'''

from _Utility import *
from sklearn.decomposition import PCA
from typing import Literal, Union
import gc
import traceback
import numpy as np


def rotate_partitions(
	#partition half of parameters
	X
	,n_partitions	:	int		=	5
	,feat_subsets	:	list	=	[]
	,partition_type	:	Literal['full_set','by_subset']='by_subset'
	,fraction_feats	:	float	=	0.5
	,no_feat_overlap:	bool	=	False
	,feats_for_all	:	list	=	[]
	#rotation half of parameters
	,rotation_type	:	Literal['PCA','Other']='PCA'
	,rotation_filter:	bool	=	False
	,filter_type	:	Literal['Retention','Count']='Retention'
	,filter_value	:	Union[float, int] = 0.5
)	->	list:
	'''
		This function is used as an easy method of executing 
		'data_partition' and 'data_rotation' with
		more readable parameters.
	Params:
	- X:
	-	_Dataset coming in, supposed to be 'X-train'.
	- n-partitions:
	-	_Number of partitions to be made from the dataset 'X'.
	- feat-subsets:
	-	_List of dictionaries of featuresets. (dict== feat-index:feat-name) 
	- partition-type:
	-	_Type of partitions made, either across all features or by featureset.
	- fraction-feats:
	-	_What fraction of features are to be included in each partition
	- no-feat-overlap:
	-	_Option to remove possibility of features being in multiple partitions.
	- feats-for-all:
	-	_Optional list to put names of any features to keep in all partitions.
	- rotation-type:
	-	_Type of rotation made on partitions 'PCA' or 'Other'.
	- rotation-filter:
	-	_Option to limit dimensionality of rotated partitions.
	- filter-type:
	-	_Type of filter used on rotation if chosen, by data or dimension retention.
	- filter-value:
	-	_Parameter for rotation filtering, fraction for 'data', integer for 'dimension'.
	'''

	#this is a no variable execution and return of both functions.

	#three comments below maps where 
	#the original X input finds use.

	#return rotated partitions from ...
	return data_rotation(
			#the partitions created here ...
			X_partitions=data_partition(
					#using original X input !!!
					X=X
					,feat_sbst=feat_subsets
					,num_parts=n_partitions
					,part_type=partition_type
					,full_excl=no_feat_overlap
					,univ_incl=feats_for_all
					,part_sbst=fraction_feats
					)
			,rotn_type=rotation_type
			,filter_=rotation_filter
			,fltr_type=filter_type
			,fltr_param=filter_value
	)

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
	X_partitions:	list							=	[]
	,rotn_type	:	Literal['PCA','Other']			=	'PCA'
	,filter_		:	bool							=	False
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
- filter-:
-	_Decide whether or not to decrease dimentionality of feature-space (post-rotation)
- fltr-type:
-	_Method of decreasing dimetionality. (Data retention (%), Dimension Count)
- fltr-param:
-	_Parameter for filtering. (Percent of data retained, Number of PC retained)
	'''
	
	#ensure X_partitions came in with any data
	if(len(X_partitions) < 1):
		print("FATAL: Some silly person decided to try and rotate ZERO partitions of data.")
		traceback.print_exc()
		raise

	#each partition now should be coming in as features ALREADY SELECTED
	# and now ready for transformation, analysis, and further selection.

	#new variable for storing PCA partitinos
	X_pca_parts = []

	#splitting up the function first by what type of rotation method is used
	match(rotn_type):

		#rotate features by primary components
		case 'PCA':
			
			#for each partition of the (training) dataset. (a subset of the dataset)
			for partition in X_partitions:
				
				#ensure partition is coming in ready to be fitted with PCA decomposer
				if(not isinstance(partition, np.ndarray)):
					traceback.print_exc()
					raise TypeError(f"FATAL: partition in X_partitions is not of type 'np.ndarray', came in as '{type(partition)}'.")
				
				#create and fit to this partition
				pca = PCA()
				pca.fit(partition)

				#set default number of components for PCA (all components of partition)
				n_components = partition.shape[1]

				#if a type of filter is declared, n_components (and nothing else) is altered in here.
				if(filter_):

					#ensuring any filter value is greater than illegal zero value
					if(fltr_param <= 0):
						traceback.print_exc()
						raise ValueError("FATAL: So why are you trying to filter components down to <= ZERO data/comps retained?")

					#filtering by data retention (in percent format)
					if(fltr_type == 'Retention'):
						#a fraction is wanted for retention, anything over 1.0 is asking for >100% data retention.
						if(fltr_param > 1.0):
							traceback.print_exc()
							raise ValueError(f"FATAL: In data_rotation, filter was of type '{fltr_type}', but value was not a fraction ({fltr_param}).")
						
						#calculate cumulative variance
						cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

						#match the number of components for data to be retained in this partition (+ 1, zero index fixer)
						n_components = np.argmax(cumulative_variance >= fltr_param) + 1

					#filter by number of features (dimensions) (in float/int format, forced to int by int(np.ceil(value)) at end)
					if(fltr_type == 'Count'):
						#a whole number is wanted for count, anything under 1.0 is asking for fraction of one component.
						if(fltr_param < 1.0):
							traceback.print_exc()
							raise ValueError(f"FATAL: In data_rotation, filter was of type '{fltr_type}', but value was not an integer ({fltr_param}).")
						
						#ensuring the number of requested primary components are less than the total number of components to pick from
						if(fltr_param > partition.shape[1]):
							traceback.print_exc()
							raise ValueError(f"FATAL: In data_rotation, {fltr_param} components were requested from {partition.shape[1]} components.")
						
						#lean towards conservative end of any complex calculation or entry (non int)
						n_components = int(np.ceil(fltr_param))

				#make formulated sized PCA
				pca = PCA(n_components=n_components)
				#append this rotated space to the 
				X_pca_parts.append(pca.fit_transform(partition))

		#other rotation types/methods. not implemented and illegal cases here.
		case 'Other':
			traceback.print_exc()
			raise NotImplementedError(f"FATAL: in data_rotation, rotating by method 'Other' is not implemented, must be by 'PCA'.")
		case _:
			traceback.print_exc()
			raise ValueError(f"FATAL: in data_rotation, rotn_type cannot be '{rotn_type}', must be 'PCA' or 'Other'.")
		
	#cleaning up and dumping no longer needed (likely large) data
	#QUICKLY QUICKLY!! 
	del X_partitions
	#GO GO GO GO GO CLEAN CLEAN CLEAN YEAAAH GOOOOOOOOOOOOOOOOOOOO
	#QUICK I NEED MY RAM TO FREE UP GET THIS DATAFRAME OUT OF HERE
	gc.collect()

	#a new list of partitions that are transformed by eigenvector matrix multiplication through PCA decomposition
	return X_pca_parts