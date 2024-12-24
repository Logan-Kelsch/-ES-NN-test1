'''
-   Data Rotating      - - Should end with an array of 'rotations' of data for n models from the train split
-   -   -   This should have a few options, pick random from all, option to keep some features in all models, ... 
-   -   -   -   ... random pick but category specific, overlapping or model unique features
'''

from _Utility import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import shuffle
from typing import Literal, Union
import gc
import traceback
import numpy as np


def rotate_partitions(
	#feature partition section of parameters
	X
	,n_feat_parts	:	int		=	5
	,feat_subsets	:	list	=	[]
	,feat_part_type	:	Literal['full_set','by_subset']	='by_subset'
	,fraction_feats	:	float	=	0.5
	,no_feat_overlap:	bool	=	False
	,feats_for_all	:	list	=	[]
	#rotation half of parameters
	,rotation_type	:	Literal['PCA','Other','None']	='PCA'
	,rotation_filter:	bool	=	False
	,filter_type	:	Literal['Retention','Count']	='Retention'
	,filter_value	:	Union[float, int] = 0.5
	#sample partition section of parameters
	,n_sample_parts	:	int		=	5
	,smpl_part_type	:	Literal['Even','Sliding']		='Even'
	,sample_shuffle	:	bool	=	False
)	->	list:
	'''
	This function is used as an easy method of executing 
		'split_by_features' and 'data_rotation' with
		more readable parameters.
	Returns: (3 items)
	-	_2D list of training sets
	-	_1D list of partition feature indices (parallel to feature-space) 
	-	_1D list of partition transformation functions (parallel to feature_space)
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

	#some comments below maps where 
	#the original X input finds use.
 
	X_feat_parts, X_part_find =split_by_features(
					#using original X input !!!
					X=X
					,feat_sbst=feat_subsets
					,num_parts=n_feat_parts
					,part_type=feat_part_type
					,full_excl=no_feat_overlap
					,univ_incl=feats_for_all
					,part_sbst=fraction_feats
					)
 
	X_feat_rots, X_part_trans=data_rotation(
			#made along the features ...
			X_partitions=X_feat_parts
			,rotn_type=rotation_type
			,filter_=rotation_filter
			,fltr_type=filter_type
			,fltr_param=filter_value
			)
 
	X_partro = split_by_samples(
		#the rotation of partitions ...
		X_partitions=X_feat_rots
		,num_parts=n_sample_parts
		,splt_type=smpl_part_type
		,shuffle=sample_shuffle
	)
 
	return X_partro, X_part_find, X_part_trans
'''
	#return sample partitions from ...
	return split_by_samples(
		#the rotation of partitions ...
		X_partitions=data_rotation(
			#made along the features ...
			X_partitions=split_by_features(
					#using original X input !!!
					X=X
					,feat_sbst=feat_subsets
					,num_parts=n_feat_parts
					,part_type=feat_part_type
					,full_excl=no_feat_overlap
					,univ_incl=feats_for_all
					,part_sbst=fraction_feats
					)
			,rotn_type=rotation_type
			,filter_=rotation_filter
			,fltr_type=filter_type
			,fltr_param=filter_value
			)
		,num_parts=n_sample_parts
		,splt_type=smpl_part_type
		,shuffle=sample_shuffle
	)
 '''

####################################################################################

def split_by_features(
	X
	,feat_sbst	:	list		=	[]
	,num_parts	:	int			=	5
	,part_type	:	Literal['full_set','by_subset']	=	'by_subset'
	,full_excl	:	bool		=	False
	,univ_incl	:	list		=	[]
	,part_sbst	:	float		=	0.5
):
	'''
		This function is used to split the data for future rotation.
		It will return a list of (training) data partitions and a list
		of the feature indices, correlated by index of each list.
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
	#array of the feature indices for each partition
	X_part_find = []

	#split function cases first by partition type:
	match(part_type):
     
		#features will be picked out of one basket
		case 'full_set':
			
			#for the number of requested partitions
			for partition in range(num_parts):
       
				#include all univeral inclusion features to each partitions features
				feature_indices = [k for k, v in all_feats.items() if v in univ_incl]
    
				#grab n TOTAL features (n minus univ OTHER features) from featureset
				feature_indices+= pick_n_from_list(num_feats - len(feature_indices), \
						[i for i in list(all_feats.keys()) if i not in feature_indices])
    
				#create new partitions for each
				X_partition.append(X[:, feature_indices])
				X_part_find.append(feature_indices)
				
		#features will be picked with consideration of feature-set
		case 'by_subset':
		
			#for each partition of requested total
			for partition in range(num_parts):
       
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
				X_part_find.append(feature_indices)

		#illegal case, neither option (of 2) entered
		case _:
			print("Illegal input for parameter 'part_type'.")
			print("Must be Literal['full_set','by_subset']")
			traceback.print_exc()
			raise

	#an array of training sets of different selections of the featurespace
	#AND an array of each partitions feature indices for applying to test data
	return X_partition, X_part_find


def data_rotation(
	X_partitions:	list							=	[]
	,X_part_find:	list							=	[]
	,rotn_type	:	Literal['PCA','Other','None']	=	'PCA'
	,filter_	:	bool							=	False
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
	#parallel list to carry all transformations made across partitions
	X_feat_trans = []

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
				X_feat_trans.append(pca)

		#in the case there is no rotation requested
		case 'None':
			
			#increasing dimensionality of arrays (to follow for program wiring)
			#while not actually doing anything at all to each partitions
			for partition in X_partitions:
				X_pca_parts.append(partition)
				#create an identity transformation as a placeholder in the list of partition transformers
				identity_transform = FunctionTransformer()
				X_feat_trans.append(identity_transform)

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

	#return list of rotated partitions of training set
	#return a list of feature transforming function for each partition
	return X_pca_parts, X_feat_trans


def split_by_samples(
	X_partitions:	list	=	[]
	,num_parts	:	int		=	5
	,splt_type	:	Literal['Even','Sliding']='Even'
	,shuffle	:	bool	=	False
)	->	list:
	'''
		This function will take a list of partitions of a dataset (partitions along featurespace, and likely rotated).
		and partition it along the samplespace as to prepare for bagging (or boosting?)
	Params:
- X-partitions:
-	_List of partitions made from the featurespace of the dataset
- num-parts:
-	_Number of partitions to make of the dataset_partitions along the samplespace
- splt-type:
-	_Method of splitting across samplespace. Even divides samples into num-parts, Sliding makes splits alike LSTM formation.
- shuffle:
-	_Option to shuffle the samples before partitioning.
	'''

	#this object will be returned. (first index: along feature space) (second index: along sample space)
	#this object is a list.
	#	this list is of partitions across the feature space of dataset X.
	#	Each partition of the feature space contains a list.
	#		This list is of partitions along the sample space.
	double_partitions = []

	#grab the length of samples in dataset, by reaching into first partition
	ds_length = len(X_partitions[0])

	#length of each partition made along sample space, floor function to leverage safe 
	# non loop function for last partition formation in next for loop
	ss_part_length = int(np.floor((ds_length / num_parts)))

	#for each feature space partition
	for fs_part in X_partitions:

		#assembly of the partitions made across sample space
		sample_parts = []

		#fs_part can be altered, but only as a temporary variable, therefore shuffle is here
		if(shuffle):
			#This can be implemented without too much hassle, just need to add y as a parameter
			#into this function to allow the y to be shuffled WITH, also need to check function
			#To know that shuffling for any segment of samples will have a consistent
			#and correct y label values to maintain truth
			raise NotImplementedError("FATAL: Shuffle has not yet been implemented into split_by_samples.")

		if(splt_type == 'Even'):
			#for making partitions, used simple p-q-r notation
			#for all sample partitions to be made except last partition, made after loop
			for i in range(num_parts - 1):
				p = i*ss_part_length
				q = (i+1)*ss_part_length
				sample_parts.append(fs_part[p:q, :])
			#final partition creation
			q = (num_parts-1)*ss_part_length
			r = ds_length
			sample_parts.append(fs_part[q:r, :])

		if(splt_type == 'Sliding'):
			#not currently implemented
			raise NotImplementedError(f"FATAL: Split type in split_by_sample of '{splt_type}' is not yet implemented.")

		#append a list of sample space partitions of this given (fs_part) feature space partition
		double_partitions.append(sample_parts)

	#returns a 2d array, with first dimension being splits along feature space, and second dimension being splits along sample space.
	return double_partitions