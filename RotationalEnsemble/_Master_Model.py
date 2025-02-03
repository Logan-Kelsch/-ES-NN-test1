'''
Intending on this file to be a collection of the modelset information 
and metamodel information to allow for full top to bottom predictions
as well as full model saving and loading.
Also looking to make this expandable to possibly level-2 model predictions,
which could be used for different time aggregation metamodels combining into one prediction?
That would be pretty advanced and likely will be built around the time I actually begin trading.
This would be the case as I should be able to collect predictions up to this point all at the same time
for on the spot personal observation of multiple models, and would desire delegating the work to a designed program.
'''

#import pyfiglet
from importlib import reload
import _Utility
#import dill
import _Neural_Net
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
from typing import Literal
import joblib
import os

class Master():
	'''
		## Overview
		The 'Master' model will be used for 
		-	Top to bottom predicting
		-	Full model saving
		-	Full model loading
		## Params
		- model-depth
		-	_The number of levels of this model
		- all-models
		-	_List of actual models by level
		- lvl0-formatters
		-	_List of lvl0-modelset specific iterables
		-	Format: `[feat-idx-info, trans-func-info]`
	'''
	def __init__(
			self
			,model_depth	:	int		=	2
			,all_models		:	list	=	[]
			,lvl0_formatters:	list	=	[]
			,lvl2_formatters:	list	=	[]
	):
		
		#Ensuring model depth is consistent with incoming data	----
		assert (model_depth == len(all_models) or len(all_models) == 0), \
			f"FATAL:, model_depth must equal len(all_models). Got ({model_depth},{len(all_models)})"
		self._model_depth	=	model_depth

		#model level declaration / seperation					----
		if(len(all_models) >= 2):
			self._level_0		=	all_models[0]
			self._level_1		=	all_models[1]
			#check for if there is a level 2 model, give None if not true
			if(len(all_models) == 3):
				self._level_2	=	all_models[2]
			else:
				self._level_2	=	None
		else:
			#the only reason there would not be added models is the intention of loading a model
			#which is where 'None' will be tested.
			self._level_0		=	None
			self._level_1		=	None
			self._level_2		=	None

		#testing what to do with formatting data, based on whether the user 
		# initiated Master with intention of loading from a directory
		if(self._level_0 == None):
			#This if case will be reached if there is no input model info, therefore intention to load model.
			self._lvl0_findx	=	None
			self._lvl0_trans	=	None
			self._lvl0_dims		=	None
		else:
			#This case is reached when there are inputted level 0&1 models
			#														----
			#ensuring feature-indices for models is consistent with modelset featuresplit size
			assert len(lvl0_formatters[0]) == len(self._level_0), \
				f"FATAL:, length of feature-indices must equal length of feature space splits in modelset.\n\
					Got ({len(lvl0_formatters[0])},{len(self._level_0[0])})"
			
			#declaration after assertion
			self._lvl0_findx	=	lvl0_formatters[0]
			if(lvl2_formatters != []):
				self._lvl2_findx	=	lvl2_formatters[0]
			
			#														----
			#ensuring feature space rotation models count is consistent with modelset featuresplit size
			assert len(lvl0_formatters[1]) == len(self._level_0), \
				f"FATAL:, length of feature space rotation functions must equal length of feature space splits in modelset.\n\
					Got ({len(lvl0_formatters[0])},{len(self._level_0[0])})"
			
			#declaration after assertion
			self._lvl0_trans	=	lvl0_formatters[1]

			#declaring dimensions of level 0 as tuple (featurespace, samplespace, modelspace)
			self._lvl0_dims		=	(len(self._level_0),len(self._level_0[0]),len(self._level_0[0][0]))

		#testing what to do with level2 formatting data based on whether
		# the user will be loading in a model or not
		if(self._level_2 == None):
			#this case is reached when the model is either to be loaded in, or when it does not contain a level2 model
			self._lvl2_findx	=	None
		else:
			#this case is reached when a level 2 model has been loaded in.
			#a formatter is required, therefore it will pull from the argument
			self._lvl2_findx	=	lvl2_formatters[0]
			'''NOTE NOTE NOTE so far only findx is implemented for lvl2 formatters, add new indices if developed further here'''

	def master_predict(self, X, threshold:float=0.5):
		'''This function is in charge of making a master prediction from a given dataset'''

		#quick assertion to stop execution if the masterclass is still NULL
		assert self._level_0 != None, f"FATAL: Master prediction blocked -> level-0 models do not currently exist."

		level_0_pred	=	self.predict_level0(X)
		print(level_0_pred.shape)#showing shape of predictions set leaving level-0

		level_1_pred	=	self.predict_level1(level_0_pred)

		if(self._model_depth>2):

			#predict using level 2 model if level 2 exists
			level_2_pred=	self.predict_level2(level_1_pred, threshold, X)
			return level_2_pred
		
		else:
			#this is reached when there is no level 2
			level_1_pred = (level_1_pred > threshold)
			return level_1_pred

	def predict_level0(self, X, model:Literal['binary','proba']='binary'):
		'''This function makes the initial modeslset predictions for the master_predict function
		## Returns
		This function returns the prediction set, as data interpretable by a model'''

		predictions = []

		for f, fspace in enumerate(self._level_0):
			for sspace in fspace:
				for model in sspace:
					#NOTE HERE we are looking at an iondividual model HERE END#NOTE

					#the local set here is the Xtest set rotated relative to each model
					#explanation:
					#Use rotation function[relevant to model].to transform(test-set using only[:, features[that are model specific]])
					local_set = self._lvl0_trans[f].transform(X[:, self.lvl0_findx[f]])
					local_pred= model.predict(local_set)
					local_pred= local_pred > 0.5
					predictions.append(local_pred)
					#proba_predictions.append(model.proba_predict())

		#return the transpose and np.array of this information since its a bunch of arrays of predictions
		if(len(predictions) < len(predictions[0])):
			predictions = np.array(predictions).T
			return np.squeeze(predictions)
		else:
			return np.array(predictions)
	
	def predict_level1(self, lvl0_predset):
		'''This function makes the level-1 / meta prediction based off of the prediction set generated by level 0
		## Returns
		This function returns a new prediction set if there is a level 2 model, OR a single prediction'''
		if(self._model_depth > 2):
			#currently only implementing a vertical stacking lvl2, meaning does nothing here!! (for now)
			pass

		#This else is reached when level 1 is the top level of the model
		y_pred = self._level_1.predict(lvl0_predset)
		return y_pred
		
	def predict_level2(self, lvl1_pred, threshold, X):
		'''this function takes the prediction from the level-2 model. Currently I am only implementing
		a vertical stacking model, specifically SVM with the focus of precision!!! 
		NOTE Consider need for modification if expansion later on in this area of code.'''

		#here, the lvl1 predictions run parallel to X coming in,
		# therefore direct preparation of X can be utilized here, followed immediately by prediction filtering.

		#this function here is redeclaring X as only the features that will be utilized in the actual lvl2 pred
		X = X[:, self._lvl2_findx]

		#format the level 1 predicitons according to threshold values
		lvl1_pred = (lvl1_pred > threshold)

		#this function is looking at where the level-1 model predicts 1
		#and then predicts using the level-2 model on those instances
		#possible error area as of 1/30/25 6:46pm unsure if np.arange follows within np.where
		#						update at  7:28pm, this version works in ipynb will use this.
		y_pred = lvl1_pred
		#alter each prediction
		for p in range(len(lvl1_pred)):
			if(y_pred[p] == 1):
				y_pred[p] = self._level_2.predict(X[p].reshape(1, -1))

		return y_pred
	
	def master_predict_fullth(self, X, y, definition:Literal['high','low','min']='low'):
		'''This function is used to show the accuracy for each threshold value'''
		ths = []
		match(definition):
			case 'high':
				d=1
			case 'low':
				d=5
			case 'min':
				d=25
		for thp in range(0, 101, d):
			#for each threshold percent, make prediction and append to list
			local_pred = self.master_predict(X, threshold=(thp/100))
			ths.append(accuracy_score(y, local_pred))
		#output as a line graph
		_Utility.plot_standard_line(ths, axhline=[ths[0],ths[-1],0.5])

	def save_model(self, name:str='tmp_model', overwrite:bool=True):
		'''This function saves the current model data into a folder of current directory.
		## Params
		- name:
		-	_Name of parent folder for current model.
		- overwrite:
		-	_Boolean option to allow overwriting of folder creation.
		## Returns
		Nothing
		'''

		#creating folder tree with this try except
		try:
			#create parent folder
			os.makedirs(name, exist_ok=overwrite)

			#create paths for each folder
			subfolder_paths = [os.path.join(name, 'level_'+str(i)) for i in range(0, self._model_depth)]

			#create subfolders
			for subfolder in subfolder_paths:
				os.makedirs(subfolder, exist_ok=overwrite)
			
			#print output for folder creation validation
			print('Folder tree generated successfully.')
		except Exception as e:
			print(f'Folder tree generated unsuccessfully:\n{e}')
		
		#level 0 model saving
		#saving models by using index values
		for f, fspace in enumerate(self._level_0):
			for s, sspace in enumerate(fspace):
				for m, model in enumerate(sspace):

					#NOTE HERE we are looking at an individual model HERE 
					#attempt to see if the model has a built in '.save' attribute.
					#This attribute is a characteristin in non-picklable models,
					#which the only one I know is tensorflow / 'tf.keras' models.
					#This should allow for any model currently implemented (1/19/25)
					#to be able to save with no issue. (sklearn aeon tf.keras) END#NOTE

					#create a path variable
					model_path = str(f'{name}/level_0/model_{f}_{s}_{m}')

					try:
						#.save attribute case, currently only tf.keras sequential models / NN class
						if(hasattr(model, 'save_as')):
							#save with attribute
							model.save_as(model_path)
							print(f'PATH: {model_path}')
							print('keras saving complete')
						jl_path = model_path+'.joblib'
						#joblib.dump(model.dump(), jl_path)
						#print('joblib saving complete')
					except Exception as e:
						print(f'Level-0 model saving to -> {model_path} could not save properly:\n{e}')

		#level 1 model saving
		#because I do not think this will be hard to implement, I will code this as only saving a single model
		#and also under the assumption that it will be a tf.keras model
		try:
			#attempting to save the level1 model as single model and assuming its a tf.keras model
			print(type(self._level_1))
			self._level_1.save(f'{name}/level_1/model_0.keras')
		except Exception as e:
			print(f'Level-1 model saving to -> {f'{name}/level_1/model_0.keras'} could not save properly:\n{e}')

		#level 2 model saving
		#because I do not think this will be hard to implement, I will code this as only saving a single model
		#and also under the assumption that it will be a joblib model
		try:
			#attempting to save the level2 model as single model and assuming its a picklable model
			joblib.dump(self._level_2, f'{name}/level_2/model_0.joblib')
		except Exception as e:
			print(f'Level-2 model saving to -> {f'{name}/level_2/model_0.joblib'} could not save properly:\n{e}')

		#all models have been saved, now the lvl0 formatting information needs to be saved.
		try:
			#attempting to save the 2 item lvl0 formatting information to a single .joblib with joblib
			formatter_path = f'{name}/level_0/lvl0_formatter.joblib'
			joblib.dump([self._lvl0_findx, self._lvl0_trans, self._lvl0_dims], formatter_path)
		except Exception as e:
			print(f'Failed to save level-0 formatters to -> {formatter_path}:\n{e}')

		#level0 formatters are saved, now the lvl2 formatting information needs to be saved
		try:
			#attempting to save the indices variable for features (MUST HAVE orig_time included) used in model
			formatter_path = f'{name}/level_2/lvl2_formatter.joblib'
			joblib.dump(self._lvl2_findx, formatter_path)
		except Exception as e:
			print(f'Failed ot save level-2 formatters to -> {formatter_path}:\n{e}')


		'''
		NOTE currently have not implemented a level-1 model length >1, nor level-1 formatting information
		NOTE If I am to implement this, need to change the saving folders names away from lvl0frmt and
			 move that data into the level_0 format, and such like that.
		NOTE the current state of this function is limited, only because I believe it will be a bit more time
			 before I expect to need a level_2, or decide that it will be worth the time.
		'''
		#all models and formatters are saved, end function. 					---------------

	def load_model(self, name:str='tmp_model', is_NN:bool=False, overwrite:bool=True):
		'''This function will load in a master model from a provided directory.
		## Params
		- name:
		-	_Name of the directory to pull the model from
		- overwrite:
		-	_Allow overwriting of the current initialized Master model (if applicable).
		## Returns
		Nothing. All model data is stored in this class.
		'''

		if(self._level_0 != None or\
	 	   self._level_1 != None):
			if(not overwrite):
				print(f"Loading of model '{name}' blocked by overwrite parameter.")

		#beginning with loading in formatting data
		try:
			#pulling out data from files
			pulling_path = f'{name}/level_0/lvl0_formatter.joblib'
			pulled_data = joblib.load(pulling_path)
			
			#loading the data into class details
			self._lvl0_findx = pulled_data[0]
			self._lvl0_trans = pulled_data[1]
			self._lvl0_dims  = pulled_data[2]
		except Exception as e:
			print(f'Could not load level 0 formatters:\n{e}')

		#now try loading in level 2 formatting data
		try:
			#pulling out data from files
			pulling_path = f'{name}/level_2/lvl2_formatter.joblib'
			if(os.path.exists(pulling_path)):
				pulled_data = joblib.load(pulling_path)
		
				#loading the data into class details
				self._lvl2_findx	=	pulled_data
		except Exception as e:
			print(f'Could not load level 0 formatters:\n{e}')

		#try loading in the level 0 modelset
		try:
			#list of extensions for easy use within nested loop
			extensions = ['.joblib','.keras']
			
			#try to load in level 0 modelset
			#this list is the splits of feature space
			model_fspace = []
			for f in range(self._lvl0_dims[0]):

				#this list is the splits of the sample space
				model_sspace = []
				for s in range(self._lvl0_dims[1]):

					#this list is the individual splits of model space
					model_mspace = []
					for m in range(self._lvl0_dims[2]):

						#each variable in this list is an individual model
						#so NOTE HERE we are looking at individual models for loading

						#according to the saving function we built this should load without fault
						pulling_path = str(f'{name}/level_0/model_{f}_{s}_{m}')

						#going to iterate through the possible extensions to load in based on model type
						for ext in extensions:
							full_path = pulling_path+ext

							#if the current iterated path exists
							if(os.path.exists(full_path)):
								
								#now we have to check what extension was accepted
								if(ext == '.joblib'):
									#this if statement is depricated with loading method
									#as of 1.21.25

									#model is joblib file, load in with built in func

									model = joblib.load(full_path)
									
									if(hasattr(model, 'load_ext')):
										#this if case is for keras to load in its model from file
										model.load_ext(full_path)

								elif(ext == '.keras'): 
									#THis case suggests that a keras model was saved
									#This will be brought in with the following method
									  #which will be standard as of 1/22/25, currently
									  #without consideration of saving/loading LSTM info
									  #or any other model retention, will be best off saving
									  #info file for loading in proper full NN class infos

									#NOTE HERE we know full_path is .keras model
									#also currently without regression compatability
									model = _Neural_Net.NN('Classification')
									#here loads in the keras model in to model.model attr
									model.load_ext(full_path)
								
								else:

									#left this here for if different models are 
									#implemented with different built in model
									#loading attributes where the model is
									#also not picklable
									raise NameError('FATAL: IMPOSSIBLE CASE')
								
								model_mspace.append(model)
								
					#begin cascade loading into 3D grid
					model_sspace.append(model_mspace)
				model_fspace.append(model_sspace)
			self._level_0 = model_fspace
		except Exception as e:
			print(f'Failed to load in level-0 modelset:\n{e}')

		#now try to load in level-1 model
		try:
			#list of extensions for easy checking and code readability
			extensions = ['.joblib','.keras']

			'''NOTE NOTE this path is a location of where more is needed for implementation
			of level-2 model usage in master class.'''

			pulling_path = f'{name}/level_1/model_0'

			#going to iterate through the possible extensions to load in based on model type
			for ext in extensions:
				full_path = pulling_path+ext

				#if the current iterated path exists
				if(os.path.exists(full_path)):
								
					#now we have to check what extension was accepted
					if(ext == '.joblib'):

						#model is joblib file, load in with built in func
						self._level_1 = joblib.load(full_path)

					elif(ext == '.keras'):

						#model is keras file, load in with built in func
						self._level_1 = load_model(full_path)
						
					else:

						#left this here for if different models are 
						#implemented with different built in model
						#loading attributes where the model is
						#also not picklable
						raise NameError('FATAL: IMPOSSIBLE CASE')

			#model is now loaded into self._level_1
		except Exception as e:
			print(f'Could not load in level-1 model:\n{e}')

		#now try to load in level-2 model
		try:

			#list of extensions for easy checking and code readability
			extensions = ['.joblib','.keras']

			pulling_path = f'{name}/level_2/model_0'

			#going to iterate through the possible extensions to load in based on model type
			for ext in extensions:
				full_path = pulling_path+ext

				#if the current iterated path exists
				if(os.path.exists(full_path)):
								
					#now we have to check what extension was accepted
					if(ext == '.joblib'):

						#model is joblib file, load in with built in func
						self._level_2 = joblib.load(full_path)

					elif(ext == '.keras'):

						#model is keras file, load in with built in func
						self._level_2 = load_model(full_path)
						
					else:
						#left this here for if different models are 
						#implemented with different built in model
						#loading attributes where the model is
						#also not picklable
						raise NameError('FATAL: IMPOSSIBLE CASE')

			#model is now loaded into self._level_2
		except Exception as e:
			print(f'Could not load in level-2 model:\n{e}')

	#model depth attribute	----

	@property
	def model_depth(self):
		return self._model_depth
	
	@model_depth.setter
	def model_depth(self, new:int):
		self._model_depth = new
	
	#level 0 attribute	----

	@property
	def level_0(self):
		return (len(self.level_0), len(self._level_0[0]), \
		  [type(model) for model in self._level_0[0][0]])
	
	@level_0.setter
	def level_0(self, new:any):
		self._level_0 = new

	#level 1 attribute	----

	@property
	def level_1(self):
		#for in case this expands to level 2, assuming only list of models under that case
		if(type(self._level_1) == list):
			return (len(self._level_1), type(self._level_1[0]))
		return type(self._level_1)
	
	@level_1.setter
	def level_1(self, new:any):
		self._level_1 = new

	#level 2 attribute	----
	
	@property
	def level_2(self):
		return type(self._level_2)
	
	@level_2.setter
	def level_2(self, new:any):
		self._level_2 = new

	#level 2 formatting attributes	----
	
	@property
	def lvl2_findx(self):
		return self._lvl2_findx
	
	@lvl2_findx.setter
	def lvl2_findx(self, new):
		self._lvl2_findx = new
	
	#here are the level 0 formatting attributes

	@property
	def lvl0_findx(self):
		return self._lvl0_findx
	
	@lvl0_findx.setter
	def lvl0_findx(self, new):
		self._lvl0_findx = new

	@property
	def lvl0_trans(self):
		return self._lvl0_trans
	
	@lvl0_trans.setter
	def lvl0_trans(self, new):
		self._lvl0_trans = new

	@property
	def lvl0_dims(self):
		return self._lvl0_dims
	
	@lvl0_dims.setter
	def lvl0_dims(self, new:tuple):
		self._lvl0_dims = new
