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
		,class_count	:	int													=	4

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

	def build_fit(
		self,
		X_train
		,y_trian
		,LSTM			:	bool												=	False
		,time_steps		:	int													=	5
		,architecture	:	Literal['default_shallow','default_deep','custom']	=	'default_shallow'
		,custom_architecture:Union[None, list]									=	None
	):
		#declare model structure as variable, possibly self.model already
		#model.compile
		#return the model?
		#no no, make it interact like sklearn/tf models.
		#This function will check or build the architecture

		#Okay, this function should create and build self.model (of type tf.keras.Sequential)
		#then it will call self.model.compile
		#then it will call self.model.fit
		#and it will then return history

		'''
		cmp = 'C'
		if tf.config.list_physical_devices('GPU'):
    	cmp = 'G'
    	pass
		with tf.device('/'+cmp+'PU:0'):
    	print('Running on: '+cmp+'PU\n')
    	model = build_LSTM_model()
    	#loaded_model = load_model()
    	used_model = model
    	history = model_fit(X_train, y_train, used_model, epochs, 
                        shuffle=True,
                        verbose=1, 
                        validation_data=(X_val, y_val), 
                        batch_size=125,
                        callbacks=[reduce_lr, early_stopping],
                        params=params)
		'''
		'''
		model.compile(optimizer=opt6,
                  loss=params.loss_function
                  ,metrics=params.performance_metrics)
		'''
		'''
		#architecture specific information, this contains the actual model
		#the actual model is a tf.keras.sequential( where here is a list of tf.keras.layers objects)
		match(architecture):
			case 'default_shallow':
				pass
			case 'default_deep':
				pass
			case 'custom':
				self.model	=	tf.keras.Sequential()
		'''
		'''
		def model_fit(X, y, model, epochs, shuffle, verbose, validation_data,\
              batch_size, callbacks, params):
    if(params.model_type == 'Regression'):
        history = model.fit(X, y, 
                epochs=epochs,
                shuffle=shuffle, 
                verbose=verbose,
                validation_data=validation_data,
                batch_size=batch_size, 
                callbacks=callbacks)
    else:   ### CLASSIFICATION  ################
        history = model.fit(X, y, 
                epochs=epochs,
                shuffle=shuffle, 
                verbose=verbose,
                validation_data=validation_data,
                batch_size=batch_size, 
                callbacks=callbacks,
                class_weight=params.class_weights)
    return history
		'''

	def format_LSTM():
		'''This function will take in a dataset and return an LSTM format version'''
		pass
	
	def save_as(self, name:str):
		'''This function will save a given model as a .keras / .xxxxx file'''
		#'model' here will have to be of type tf.keras.Sequential
		self.model.save('name')
