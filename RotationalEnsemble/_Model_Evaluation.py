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
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
#import libraries that will contain used model type functions - for .predict()
from sklearn.tree import DecisionTreeClassifier
from aeon.classification.sklearn import RotationForestClassifier

'''Iterations of models should run along iterations of X,y first two dimensions'''

def evaluate_models(
		models
		,X_findx
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
	- X-findx:
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

			seen_performances = [] #'seen'  		training data relative to each model
			unsn_performances = [] #'unseen'		training data relative to each model
			indp_performances = [] #'independent'	test data transformed for each model
			#iterate through the 3D array of models. (fspace=featurespace, sspace=samplespace)
			for f_index, a_given_fspace in enumerate(models):
				for s_index, a_given_sspace in enumerate(a_given_fspace):
					for m_index, model in enumerate(a_given_sspace):

						'''NOTE HERE we are looking at a specific model at location (f,s,m) in models HERE NOTE'''
						#NOTE refer to function argument definitions for any indexing questions.

						#if user has requested self test for all models
						if(test_whch in ('train','all')):

							#all data in X_train is pre-transformed, so X_train[f][s] is ready to go!
							local_y_pred = model.predict(X_train[f_index][s_index])

							#collect all datapoints with parallel targets at y_train[s]
							accuracy = accuracy_score(	y_train[s_index], local_y_pred)
							precision= precision_score(	y_train[s_index], local_y_pred)
							recall   = recall_score(	y_train[s_index], local_y_pred)
							conf_matx= confusion_matrix(y_train[s_index], local_y_pred)

							#turn all data into a tuple to add to the flatten performance list.
							local_performance = (f_index, s_index, m_index, accuracy, precision, recall, conf_matx)
							seen_performances.append(local_performance)

						#if user requests to test unseen section of training data
						if(test_whch in ('unseen_train','all_unseen','all')):
							
							#create a flat list of predictions variable
							local_y_pred = []

							#create a custom target list for performance evaluation
							local_y_true = []

							#iterate through all X_train unseen
							for f, fspace in enumerate(X_train):
								for s, train_subset in enumerate(fspace):
									#if the training subset is not the models training data (if is unseen data)
									if(f!=f_index|s!=s_index):
										local_y_pred.extend(model.predict(train_subset))
										local_y_true.extend(y_train[s])

							#collect all datapoints with parallel targets at y_train[s]
							accuracy = accuracy_score(	local_y_true, local_y_pred)
							precision= precision_score(	local_y_true, local_y_pred)
							recall   = recall_score(	local_y_true, local_y_pred)
							conf_matx= confusion_matrix(local_y_true, local_y_pred)

							#turn all data into a tuple to add to the flatten performance list.
							local_performance = (f_index, s_index, m_index, accuracy, precision, recall, conf_matx)
							unsn_performances.append(local_performance)


						#if user requests independent test for all models
						if(test_whch in ('independent','all')):
							
							#bring in the test set, transform according to current model
							#X_trans is the list of feature transforming functions (ex: pca)
							#X_trans[f] is an actual transforming function
							#X_test[:, findx] is the testing featureset of the current models featurespace
							local_indp = X_trans[f_index].transform(X_test[:, X_findx[f_index]])

							#local_independent set is now ready for prediction on current model
							#predict transformed X test set
							local_y_pred = model.predict(local_indp)

							#collect all datapoints with parallel targets y_test
							accuracy = accuracy_score(	y_test, local_y_pred)
							precision= precision_score(	y_test, local_y_pred)
							recall   = recall_score(	y_test, local_y_pred)
							conf_matx= confusion_matrix(y_test, local_y_pred)

							#turn all data into a tuple to add to the flatten performance list.
							local_performance = (f_index, s_index, m_index, accuracy, precision, recall, conf_matx)
							indp_performances.append(local_performance)
			
			#Here we have collected all information and have 3 lists of performances recorded

			#output modelset dimensionality
			print(f"Dimensions of Trained Models:\n\tFeature Space: {len(models)}\n\tSample Space: {len(models[0])}\n\tModel Space: {len(models[0][0])}\n")

			if(len(seen_performances)>0):
				print(f"\tDisplaying all performances for all 'seen' traning samples: ({len(seen_performances)} cases)\n")
				display_evaluation(seen_performances, prfm_gnrl, prfm_stat, disp_mthd)
			if(len(unsn_performances)>0):
				print(f"\n\tDisplaying all performance for all 'unseen' training samples: ({len(unsn_performances)} cases)\n")
				display_evaluation(unsn_performances, prfm_gnrl, prfm_stat, disp_mthd)
			if(len(indp_performances)>0):
				print(f"\n\tDisplaying all performances for all independent samples: ({len(indp_performances)} cases)\n")
				display_evaluation(indp_performances, prfm_gnrl, prfm_stat, disp_mthd)

#This function will be a subfunction used in 'evaulate_models' in this file.
#This will be used specifically to output performances as requested.
def display_evaluation(
		prfm_tupls	:	list	=	[]
		,prfm_gnrl	:	Literal['by_each','by_mspace','by_sspace','by_fspace','by_set','all']	=	'all'
		,prfm_stat	:	Literal['avg','std_dev','high_low','all']	=	'all'
		,disp_mthd	:	Literal['as_graph','as_value','as_both']	=	'as_both'
):
	'''
	This function is used within evaluate-models as an easier method of displaying performances.
		Params:
	- prfm-tupls:
	-	_performance tuples, a list of tuples of model's basic and performance info.
	-	_format: (feature-space index, sample-space index, model-space index
	-				_lmld accuracy, lmld precision, lmld recall, lmld confusion matrix)
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
		Definitions:
	- lmld:
	-	_local model on local dataset.
	'''

	#this parameter will be used as the universal list of all generalizations..
	#a splitting area will be appended to this variable for each generalization included.
	#for each setting, there is only one generalization, so there is only more than one item in
	#	original variable length here when 'all' is selected, it will then be length 5.

	#	NOTE NOTE DEFINITION: EIC = 'each item containing' END#NOTE END#NOTE
	#the 'splitting area' will just be a list of sublists, making this under all circumstances
	# a list EIC splitting-lists EIC raw-performance-sublists
	performance_generalizations = []

	#first split up performances based on 'Generalization' parameter
	
	if(prfm_gnrl in ('by_each','all')):
		#if evaluation is requested to be done on the most broad level, evaluated evenly across all models
		#create dummy list to bypass splitting area in 'performance_generalizations' as there are no sublists
		list_nester = []
		list_nester.append(prfm_tupls)
		#add to generalizations list
		performance_generalizations.append(list_nester)

	if(prfm_gnrl in ('by_fspace','all')):
		#if evaluation is requested to be done based off of featurespaces of the modelset
		
		#create a grouping variable
		fgroups = defaultdict(list)
		
		#append each performance collected based off of featurespace index
		for individual_performance in prfm_tupls:
			#like bucket system, use fspace index value AS indexing value for append location!
			fgroups[individual_performance[0]].append(individual_performance)
		
		#NON-overwriting method of turning defaultdict into a list of lists
		fspace_grouped = list(fgroups.values())
		
		#this variable being appended is now a list of performance sublists, AKA a splitting-list
		performance_generalizations.append(fspace_grouped)

	if(prfm_gnrl in ('by_sspace','all')):
		#if evaluation is requested to be done based off of samplespaces of the modelset

		#create a grouping variable
		sgroups = defaultdict(list)

		#append each performance collected based off of samplespace index
		for individual_performance in prfm_tupls:
			#like bucket system, use sspace index values AS indexing value for append location!
			sgroups[individual_performance[1]].append(individual_performance)
		
		#NON-overwriting method of turning default dict into a list of lists
		sspace_grouped = list(sgroups.values())

		#this variable being appended is now a list of performance sublists, AKA a splitting-list
		performance_generalizations.append(sspace_grouped)

	if(prfm_gnrl in ('by_mspace','all')):
		#if evaulation is requested to be done based off of modelspaces of the modelset

		#create a grouping variable
		mgroups = defaultdict(list)

		#append each perfromance collected based off of modelspace index
		for individual_performance in prfm_tupls:
			#like bucket system, use mspace index values AS indexing value for append location!
			mgroups[individual_performance[2]].append(individual_performance)

			#NON-overwriting method of turning default dict into a list of lists
			mspace_grouped = list(mgroups.values())

			#this variable being appended is now a list of performance sublists, AKA a splitting-list
			performance_generalizations.append(mspace_grouped)

	if(prfm_gnrl in ('by_set','all')):
		#going to implement this later..
		pass

	#second collect all performance statistics requested, and flatten into tuple
	#Consists of: Average, StdDev, High, Low

	#This variable will be used to store statistics in same format as performance_generalizations.
	performance_statistics = []


	#need to check each generalization, create splitter list by iterating each item there
	for generalization in performance_generalizations:

		#an appending list to add each split to to append it accordingly to performance statistcs list
		general_stats = []

		for split in generalization:
			#NOTE HERE we are looking at a list of performances already sectioned out HERE END#NOTE

			#in mirrored format, the split is replaced with 2D array
			#	list of [accuracy data, precision data, recall data, combined confusion matrix(implement later)]
			# (tuple indices:)	   3			4			  5				      6
			#	each acc,prec,rec will be a list of [avg, std, low, high]
			local_accuracy  = [prfm[3] for prfm in split]
			local_precision = [prfm[4] for prfm in split]
			local_recall	= [prfm[5] for prfm in split]

			#This is the item that will be appended for EACH split, building a splitting-list
			split_stats = []
			
			#collect all desired statistics
			accuracy_stats 	=[np.mean(local_accuracy), np.std(local_accuracy), np.min(local_accuracy), np.max(local_accuracy)]
			precision_stats =[np.mean(local_precision),np.std(local_precision),np.min(local_precision),np.max(local_precision)]
			recall_stats	=[np.mean(local_recall),   np.std(local_recall),   np.min(local_recall),   np.max(local_recall)]
			#not currently implementing the confusion matrix
			#combined_confusion matrix = bada bing bada boom

			#append to create list of split specific statistical points
			split_stats.append(accuracy_stats)
			split_stats.append(precision_stats)
			split_stats.append(recall_stats)
			#not currently implementing the confusion matrix
			#split_stats.append(combined_confusion_matrix)

			#each split data goes here, general_stats because a splitting-list
			general_stats.append(split_stats)
		#append each splitting list to the performance statistics, mirroring structure of performance generalizations
		performance_statistics.append(general_stats)

	#third display performances based on 'Display Method' parameter	
	#iterating method will be replicated between both of these options.

	#iterable list for ease of titling!
	generalization_titles = ['Overall','By Feature Space','By Sample Space','By Model Space','TITLE ERROR SET CASE IS NOT IMPLEMENTED']

	#iterable list of ease of titling!
	statistics_titles = ['Accuracy','Precision','Recall','TITLE ERROR CONFUSION MATRIX CASE IS NOT IMPLEMENTED']

	'''NOTE CURRENT	VALUE DISPLAY INDENTATION (USE OF \t) IS BUILT SPECIFICALLY FOR 4 SPACE TABS. END NOTE'''

	if(disp_mthd in ('as_value','as_both')):

		#iterate through each generalization selected
		for g_index, generalization in enumerate(performance_statistics):
			print(f'\nStatistics {generalization_titles[g_index]}:\n')

			#NOTE here we are looking at each generalization, where 'generalization is a splitting-list
			#SO we are going to show the statistics of all splits! (meaning of all occupied spaces of given dimension)
			for split_index, split in enumerate(generalization):

				#NOTE here we are looking at each split statistics, where split is iterable stat categories (acc,prec,rec)
				print(f'\tSplit {split_index}:')
				print(f'\t\t\t\t\tAverage\tStd.Dv.\tLow\t\tHigh')
				for stat_index, stat_cat in enumerate(split):
					#This print line will output the statistic type, idents extra for rec, then prints out all four statistics for each type (a,s,l,h)
					print(f"\t\t{statistics_titles[stat_index]}{'\t' if stat_index==2 else ''}\t{stat_cat[0]}\t{stat_cat[1]}\t{stat_cat[2]}\t{stat_cat[3]}")


	if(disp_mthd in ('as_graph','as_both')):
		raise NotImplementedError(f'FATAL: The output of graphs has not yet been implemented.')