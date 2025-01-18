'''
	NOTE NOTE NOTE 
	This file will contain all of the custom neural network code for any given implementation.
	Going to try to hopefully actually allow for all LSTM to be done within this file???
	If so that would allow for much more expandability with simple implementation of lstm testing and training,
	especially if we can go back and make sure all custom train/test functions are passed split data,
	making anything where LSTM is built here non-overlapping no matter what.

	NOTE delete this once this is considered and fulfilled
'''

import tensorflow as tf
from typing import Literal, Union



#This class is for a traditional neural network, using tensorflow
class NN:
	def __init__(
		self
		,predict_mode	:	Literal['Classification','Regression','Multi-Class']=	'Classification'
		,architecture	:	Literal['default_shallow','default_deep','custom']	=	'default_shallow'
		,LSTM			:	bool												=	False
		,time_steps		:	int													=	5
		,class_count	:	int													=	4
		,custom_architecture:Union[None, list]									=	None

	):
		self.pred_mode	=	predict_mode
		
		#prediction mode specific characteristics
		match(predict_mode):
			case 'Classification':
				self.target_activation	=	'sigmoid'
				self.target_neurons		=	1
				self.performance_metrics=	['precision','recall','accuracy']
				self.monitor_parameter	=	'val_accuracy'
				self.monitor_condition	=	'max'
				self.loss_function		=	'binary_crossentropy'
			case 'Multi-Class':
				self.target_activation	=	'softmax'
				self.target_neurons		=	class_count
				self.performance_metrics=	['precision','recall','accuracy']
				self.monitor_parameter	=	'val_accuracy'
				self.monitor_condition	=	'max'
				self.loss_function		=	'categorical_crossentropy'
			case 'Regression':
				self.target_activation	=	'linear'
				self.target_neurons		=	1
				self.performance_metrics=	['R2Score','root_mean_squared_error']
				self.performance_metrics=	'val_mse'
				self.monitor_condition	=	'min'
				self.loss_function		=	'mse'

		#Learning rate reduction function 
		self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    		monitor=self.monitor_parameter,
    		mode=self.monitor_condition,
    		factor=0.75, 
    		patience=1000, 
    		min_lr=1e-6
		)
		
		#architecture specific information, this contains the actual model
		#the actual model is a tf.keras.sequential( where here is a list of tf.keras.layers objects)
		match(architecture):
			case 'default_shallow':
				pass
			case 'default_deep':
				pass
			case 'custom':
				self.model	=	tf.keras.Sequential()
	
	#Here are a couple property functions, will probably miss a few and will need to reup this frequently

	@property
	def target_activation(self):
		return self.target_activation

	@property
	def target_neurons(self):
		return self.target_neurons
	
	@property
	def performance_metrics(self):
		return self.performance_metrics

	@property
	def monitor_parameter(self):
		return self.monitor_parameter

	@property
	def monitor_condition(self):
		return self.monitor_condition

	@property
	def loss_function(self):
		return self.loss_function
	
	'''This set of 3 characteristics may be buggy, before testing, unsure of ease of accessing these variables'''
	
	@property
	def reduce_lr_factor(self):
		return self.reduce_lr.factor
	
	@property
	def reduce_lr_patience(self):
		return self.reduce_lr.patience
	
	@property
	def reduce_lr_min_lr(self):
		return self.reduce_lr.min_lr
	
	#Here are the setter functions twinning the property functions, also will need reup with updates

	@target_activation.setter
	def target_activation(self, new:str):
		self.target_activation = new

	@target_neurons.setter
	def target_neurons(self, new:int):
		self.target_neurons = new
	
	@performance_metrics.setter
	def performance_metrics(self, new:list):
		self.performance_metrics = new

	@monitor_parameter.setter
	def monitor_parameter(self, new:str):
		self.monitor_parameter = new

	@monitor_condition.setter
	def monitor_condition(self, new:str):
		self.monitor_condition = new

	@loss_function.setter
	def loss_function(self, new:str):
		self.loss_function = new

	'''This set of 3 characteristics may be buggy, before testing, unsure of ease of accessing these variables'''
	
	@property
	def reduce_lr_factor(self):
		return self.reduce_lr.factor
	
	@property
	def reduce_lr_patience(self):
		return self.reduce_lr.patience
	
	@property
	def reduce_lr_min_lr(self):
		return self.reduce_lr.min_lr
	
	

	#Here are other functions that need to be fully implemented.
	#As im building the structure of this im just going to write empty functions that will need completed for future use



	def format_LSTM():
		'''This function will take in a dataset and return an LSTM format version'''
		pass
	
	def save():
		'''This function will save a given model as a .keras / .xxxxx file'''
		pass
