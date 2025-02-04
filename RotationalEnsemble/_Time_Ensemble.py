
'''
This file will be used for loading in a few different master models
and then collecting the popular vote for given samples.
The main difference between the master models is that they should be predicting different times in the future
'''

import _Master_Model
from typing import Literal
import numpy as np

class Chronos():
	'''
	### In this function Im going to load in the test set within the function.
	### I am then going to load in a seperate set of just the features,
	-	This should be used by all models
	I am then going to load in a set of ALL requested targets,
	-	find seperable pattern in csv for their names so
	-	the master-names variable can rip up target-set.
	-	in preprocessdata, implement targetset and also index 0+comparative only
	### This target set should be indices parallel to master-names and loaded models
	### Once all models are loaded in, all will predict and a set identical to lvl0predset format will be returned
	### Make all master models depth 3, try pop vote then decide if its worth making deeper model (+target)
	after this the last thing to implement is final function (raw thinkorswim export into prediction.)
	- looks like: load_raw_to_fullset, chronos_predict, prediction_visualization then risk management development.
	
	this class shoyld only be implemented if we will make a 
	saveable chronos model. but popular vote makes the most
	sense at the moment, so use chronos_predict first
	'''
	def __init__(
		master_names    :   list    =   []
		,loading_kwargs	:	dict	=	{}
):
		raise NotImplementedError()
	
	def predict():
		raise NotImplementedError()
	
	def fit():
		raise NotImplementedError()
	
# NOTE NOTE end of class declaration END NOTE END NOTE


def chronos_predict(
	X
	,master_names : list = []
	,loading_kwargs : dict = {}
):
	'''This function will take in a set to be trained
		and make a prediction with chosen fusion method.
			Params:
		- X:
		-   Test set, each master will do all needed transformations.
		- loading-kwags:
		-   Not sure if this is needed, for NN/master params.
		- fusion-method:
		-   Method of fusing model predictions, currently only implemented PV
		-   Due to no clear target, consider making a more universal target.
		-	mv: Minimum Vote, a given number of votes must be satsified for voting 'True'
	'''
	
	#variables are for prediction collection and model loading
	master_predictions = []
		#currently redundant data collection, may be useful NOTE in a class
		#master_models = []
	
	#load in all master models into list of masters
	for model, model_name in enumerate(master_names):
		local_master = _Master_Model.Master(model_depth=3)
		local_master.load_model(model_name)
			#currently redundant line, no need to save models for this function
			#master_models.append(local_master)
		print(f'Chronos: Predicting on Model #{model+1} ({model_name})')
		master_predictions.append(local_master.master_predict(X))
		
	#create the transpose of this, as everything is backwards
	master_predictions = np.array(master_predictions).T
	master_predictions = np.squeeze(master_predictions)
	
	#this commit is all coded on my phone, this code should not be working
	#and should be completed on my computer, but the idea
	#should get across fully.
	return master_predictions
	
	

def chronos_fusion(
	master_predictions	:	any	=	None
	,fusion_method:Literal['pv','mv']='pv'
	,vote_var	:	int	=	0
):
	#combine predictions based on method
	match(fusion_method):
	
		#if the user requests use of popular vote
		case 'pv':
			final_predictions = np.array([])
			for sample in master_predictions:
				pred_counts = np.bincount(sample.flatten())
				final_predictions = np.append(final_predictions, np.argmax(pred_counts))

		case 'mv':
			final_predictions = np.array([])
			for sample in master_predictions:
				pred_counts = np.bincount(sample.flatten())
				final_predictions = np.append(final_predictions, pred_counts[0] <= (len(master_predictions[0])-vote_var))
			
		#for any other request outside of this switch case, it has not yet been implemented.
		case _:
			raise NotImplementedError(f'FATAL: Fusion type "{fusion_method}" is not implemented')
	
	return final_predictions
