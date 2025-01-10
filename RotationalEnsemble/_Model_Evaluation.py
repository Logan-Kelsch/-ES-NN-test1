'''
-   Model evaluation
-   -   -   Copy over the printout functions of previous versions
-   -   -   consider adding some sort of proba printouts?
'''

#NOTE WHEN REACHING INTO FEATURE INDICES USED AND ROTATION DECOMPOSER USED, THAT ARE SAVED IN
#NOTE THE PARALLEL ARRAY OF X_FEAT_FIND REMEMBER THAT IT IS A 1D ARRAY, AND IT ONLY RUNS PARALLEL 
#NOTE TO THE FEATURESPACE DIMENSION AND NOT THE SAMPLESPACE DIMENSION OR MODELSPACE DIMENSION, 
#NOTE SO REFER ONLY TO THE INDEX OF THE FEATURE DIMENSION WHEN REFERING TO IT.

'''
	watching a podcast and coding so just going to write notes.

	NOTE create a self test and independent test 3d array of performances for each model type
	NOTE predicament: how to show the overall performance of models?
	NOTE	-	options to show all performances
	NOTE	-	show full averages, std deviations, high and low
	NOTE	-	show type averages, std deviations, high and low (types: featurespace, samplespace, modelspace based)
	NOTE	-	show feat-samp pair averages, std devs, hi n' lo (defin: for each training set)

'''

from typing import Literal
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
#import libraries that will contain used model type functions - for .predict()
from sklearn.tree import DecisionTreeClassifier
from aeon.classification.sklearn import RotationForestClassifier

'''Iterations of models should run along iterations of X,y first two dimensions'''

def evaluate_models(
		models
		,X_find
		,X_trans
		,X_train
		,y_train
		,X_test
		,y_test
		,prfm_gnrl	:	Literal['by_each','by_mspace','by_sspace','by_fspace','by_set','all']	=	'all'
		,prfm_stat	:	Literal['avg','std_dev','high_low','all']	=	'all'
		,disp_mthd	:	Literal['as_graph','as_value','as_both']	=	'as_both'
		,test_whch	:	Literal['independent','train','unseen_train','all_unseen','all']	=	'all'
		,pred_type	:	Literal['regression','classification']		=	'classification'
):
	'''
	This function will take a 3D array of trained models and evaluate them on a given dataset.
	It will then output desired statistics on desired generalizations.
	
		Params:

	- models:
	-	_3D array of models (D1:featurespace, D2:samplespace, D3:modelspace)
	- X-find:
	-	_1D array of feature-index lists for model specific feature grouping. (Iterates parallel to D1 of models)
	- X-trans:
	-	_1D array of find partition specific transformer functions. (Iterates parallel to D1 of models)
	- X-train:
	-	_2D array of feature/samplespace partition training sets. (Iterates parallel to D1,D2 of models)
	- y-train:
	-	_1D array of samplespace partitions of targets. (Iterates parallel to D2 of models)
	- X-test:
	-	_Feature half of dataset, for .predict of models
	- y-test:
	-	_Target half of dataset, for performance evaluation
	- prfm-gnrl:
	-	_Performance Generalization, choice for how to generalize collected performances.
	-	by-each:    _Shows individual model performance. (only supports numerical outputs in disp-mthd)
	-	by-mspace:  _Shows model performance by each model type.  (modelspace)
	-	by-sspace:  _Shows model performance by each sample set.  (samplespace)
	-	by-fspace:  _Shows model performance by each feature set. (featurespace)
	-	by-set:     _Shows model performances by training set. (feature-sample pairs)
	-	all:        _Shows all model performance generalization types.
	- prfm-stat:
	-	_Performance Statistics, show selected performance statistics.
	-	avg:        _Show average performance of model generalization.
	-	std-dev:    _Show standard deviation of model generalization.
	-	high-low:   _Show best and worst performance of model generalization.
	-	all:        _Show all statistics for each model generalization.
	- disp-mthd:
	-	_Display Method, method of displaying selected statistics of model generalizations
	-	as-graph:   _Show visualizations of statistics only. 
	-	as-value:   _Show numerical values of statistics only.
	-	as-both:    _Show both numerical values and visualizations of statistics.
	- test-whch:
	-	_Choose what sets the models will attempt to predict.
	-	independent:    _Score the models on a dataset fully disconnected from training set.
	-	train:          _Score the models on the dataset (specific samples and features) they were trained on.
	-	unseen-train:   _Score the models on the sections of the training dataset never seen by each model.
	-	all-unseen:     _Score the models on independed plus unseen-train data.
	-	all:            _Score the models on seen and unseen data (unseen true-averaged).
	- pred-type:
	-	_Declare what type of predictions the model is making, regression or classification.

		Definitions:
	- true-averaged:
	-	Averaging two sets of performances by accounting for number of samples to ensure equality of samples.
	'''
	

	'''NOTE(D1:featurespace, D2:samplespace, D3:modelspace)NOTE'''

	#begin function by type of prediction being made
	match(pred_type):

		#if models are predicting a regression value
		case 'regression':
			raise NotImplementedError(f'FATAL: Prediction Type ({pred_type}) not yet implemented in "evaluate_models"')
		
		#if models are predicting classifications
		case 'classification':
			'''
			First, collect the data in an array/list format of some sort of tuple-like datastructure.
				Tuple should be the accuracy, and the index for the feature,sample,model space of the model.
			With this, all accuracies are collected, and all performances are seperable with index variables.
			'''
			'''NOTE THINK ABOUT BRINGING IN AND INCORPORATION OF:
					-	MODEL SPECIFIC FEATURE INDICES FOR TRANSFORMATION 
					-	TRANSFORMING FUNCIONS 
					-	MODEL SPECIFIC SAMPLE INDICES
			'''

