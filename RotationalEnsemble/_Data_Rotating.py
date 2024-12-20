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

from typing import Literal, Union

def data_partition(
	num_parts	:	int								=	5
 	,part_type	:	Literal['full_set','by_subset']	=	'by_subset'
	,feat_sbst	:	list							=	[]
	,full_excl	:	bool							=	False
	,univ_incl	:	list							=	[]
	,part_sbst	:	float							=	0.5
)	->	list:
    '''
		This function is used to split the data for future rotation.
		It will return a list of (training) data partitions.
  Params:
- num-parts:
-	_Number of partitions made.
- part-type:
-	_Allows user to partition randomly through entire featurespace, or randomly and equally through each feature-subset-space.
- feat-sbst: 
-	_List of feature-subsets (as dict of df-index:feature-name)
- full-excl: 
-	_Option to disable ANY feature overlap across partitions.
- univ-incl: 
-	_List of feature names that will be included in all partitions. 
- part_sbst:
-	_The fraction (%) of each feature-subset that is included in each partition.
    '''
    
    
    return #an array of training sets of different selections of the featurespace


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