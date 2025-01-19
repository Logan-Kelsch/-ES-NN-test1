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

import pyfiglet

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
	):
		
		#Ensuring model depth is consistent with incoming data	----
		assert model_depth == len(all_models), \
			f"FATAL:, model_depth must equal len(all_models). Got ({model_depth},{len(all_models)})"
		self._model_depth	=	model_depth

		#model level declaration / seperation					----
		self._level_0		=	all_models[0]
		self._level_1		=	all_models[1]

		#optional level2 for future use, likely will not fully implement.
		if(model_depth>2):
			self._level_2	=	all_models[2]
		else:
			self._level_2	=	None

		#														----
		#ensuring feature-indices for models is consistent with modelset featuresplit size
		assert len(lvl0_formatters[0]) == len(self._level_0[0]), \
			f"FATAL:, length of feature-indices must equal length of feature space splits in modelset.\n\
				Got ({len(lvl0_formatters[0])},{len(self._level_0[0])})"
		
		#declaration after assertion
		self._lvl0_findx	=	lvl0_formatters[0]
		
		#														----
		#ensuring feature space rotation models count is consistent with modelset featuresplit size
		assert len(lvl0_formatters[1]) == len(self._level_0[0]), \
			f"FATAL:, length of feature space rotation functions must equal length of feature space splits in modelset.\n\
				Got ({len(lvl0_formatters[0])},{len(self._level_0[0])})"

		#declaration after assertion
		self._lvl0_trans	=	lvl0_formatters[1]

	#This is the top to bottom prediction
	def master_predict(self, X):
		level_0_pred	=	self.predict_level0(X)
		level_1_pred	=	self.predict_level1(level_0_pred)
		if(self._model_depth>2):
			raise NotImplementedError(f"FATAL: in master_predict, model depth is read as >2, level2 is not implemented.")
		else:
			return level_1_pred

	def predict_level0(self, X):
		prediction_set	=	any#copy how i turned the model grid predictions into dataset, return pred_dataset
		return prediction_set
	
	def predict_level1(self, lvl0_predset):
		if(self._model_depth > 2):
			pass#collect a list of predictions form here for level2prediction
		#This else is reached when level 1 is the top level of the model
		else:
			y_pred = self._level_1.predict(lvl0_predset)
			return y_pred
		
	def predict_level2(self, lvl1_predset):
		raise NotImplementedError(f"Fatal: Level 2 prediction was requested, but has not been implemented.")


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
		return None
	
	@level_2.setter
	def level_2(self, new:any):
		self._level_2 = new
	

