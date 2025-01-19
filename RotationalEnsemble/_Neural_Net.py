'''
	NOTE NOTE NOTE 
	This file will contain all of the custom neural network code for any given implementation.
	Going to try to hopefully actually allow for all LSTM to be done within this file???
	If so that would allow for much more expandability with simple implementation of lstm testing and training,
	especially if we can go back and make sure all custom train/test functions are passed split data,
	making anything where LSTM is built here non-overlapping no matter what.

	NOTE delete this once this is considered and fulfilled
'''

from _Utility import *
import tensorflow as tf
import numpy as np
from keras.callbacks import EarlyStopping
from typing import Literal, Union



#This class is for a traditional neural network, using tensorflow
class NN:
	def __init__(
		self
		,predict_mode	:	Literal['Classification','Regression','Multi-Class']=	'Classification'
		,class_count	:	int													=	4

	):
		self.pred_mode		=	predict_mode
		self.model			=	None
		self.optimizer		=	tf.keras.optimizers.Adam(learning_rate=0.001)
		self.class_weight	=	None
		self.epochs			=	None
		self.batch_size		=	None
		self.LSTM			=	None
		
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
	
	#some more properties, no setter function because they are overwritten in build_fit
	@property
	def epochs(self):
		return self.epochs
	
	@property
	def batch_size(self):
		return self.batch_size
	
	@property
	def LSTM(self):
		return self.LSTM
	
	'''
		Here were some of the optimizers I uses for NN_1 through NN_6
	opt1 = SGD(learning_rate=0.0001)
	opt2  = tf.keras.optimizers.Adam(learning_rate=0.001)
	opt3 = SGD(learning_rate=lr_schedule)
	opt4 = SGD(learning_rate=0.001, momentum=0.9)
	opt5 = tf.keras.optimizers.Adam(learning_rate=0.0005)
	opt6 = AdamW(learning_rate=0.01, weight_decay=0.00)
	'''	

	#Here are other functions that need to be fully implemented.
	#As im building the structure of this im just going to write empty functions that will need completed for future use

	def build_fit(
		self,
		X_train
		,y_train
		,X_test
		,y_test
		,epochs			:	int													=	25
		,batch_size		:	int													=	32
		,shuffle_train	:	bool												=	True
		,LSTM			:	bool												=	False
		,time_steps		:	int													=	5
		,architecture	:	Literal['default_shallow','default_deep','custom']	=	'default_shallow'
		,custom_architecture:Union[None, list]									=	None
		,optimizer_type	:	Literal['SGD','Adam','AdamW']						=	'Adam'
		,optimizer_kwarg:	dict												=	{'learning_rate':0.001}
		,train_verbose	:	int													=	1
		,test_verbose	:	int													=	1
		,rlr_factor		:	float												=	0.75
		,rlr_patience	:	int													=	1000
	):
		'''This function builds the model in NN.model, calls .compile and .fit and returns the training history'''

		#this function should create and build self.model (of type tf.keras.Sequential)
		#then it will call self.model.compile
		#then it will call self.model.fit
		#and it will then return history

		#simple method of ensuring and printing which device will be used for computation
		cmp = 'C'
		if(tf.config.list_physical_devices('GPU')):
			cmp = 'G'
		with tf.device('/'+cmp+'PU:0'):
			print('Running on: '+cmp+'PU\n')

			'''NOTE NOTE NOTE BEGIN ARCHITECTURE DEVELOPMENT END NOTE END NOTE END NOTE'''

			match(architecture):
				case 'default_shallow':
					seq = [tf.keras.layers.Input(shape= (X_train.shape[1],X_train.shape[2]) if LSTM else (X_train.shape[1]))]
					#adding a single hidden layer of neuron size -> 
					# 							lowest power of 2 >= size of featurespace coming in
					if(LSTM):
						seq.append(tf.keras.layers.LSTM(2**np.ceil(np.log2(X_train.shape[2])), return_sequences=False))
					else:
						seq.append(tf.keras.layers.Dense(2**np.ceil(np.log2(X_train.shape[2])), activation='relu'))
					#adding the output layer
					seq.append(tf.keras.layers.Dense(self.target_neurons, self.target_activation))
					#combine in to actual keras interpretable information
					self.model = tf.keras.Sequential(seq)
				case 'default_deep':
					#The default deep model will be of an exponential-tree shape, 
					#where hidden layer 1 is of size 1.5 * 2^ceiling(logbase2(# features)) - 3
					seq = [tf.keras.layers.Input(shape= (X_train.shape[1],X_train.shape[2]) if LSTM else (X_train.shape[1]))]
					
					#initial depth of model, neuron length of first hidden layer, and 2 * depth_init is non-input neuron volume of model
					depth_init = np.ceil(np.log2(X_train.shape[2]))

					#adding a single hidden layer of neuron size -> 
					# 							lowest power of 2 >= size of featurespace coming in
					if(LSTM):
						seq.append(tf.keras.layers.LSTM(2**depth_init, return_sequences=False))
					else:
						seq.append(tf.keras.layers.Dense(2**depth_init, activation='relu'))
					
					#appending connective section between first large layer and rest of hidden section
					seq.append(tf.keras.layers.Dropout(0.25))
					seq.append(tf.keras.layers.BatchNormalization())
					
					#add remaining layers here, skipping every other power, found it near redundant in my experience
					for layer_size in range(depth_init-1, self.target_neurons+1, -1):
						seq.append(tf.keras.layers.Dense(2**layer_size, activation='relu'))
						seq.append(tf.keras.layers.Dropout(0.25))
						seq.append(tf.keras.layers.BatchNormalization())

					#adding the output layer
					seq.append(tf.keras.layers.Dense(self.target_neurons, self.target_activation))

					#combine in to actual keras interpretable information
					self.model = tf.keras.Sequential(seq)

				case 'custom':
					#no work to do here, anything custom comes in as list fully built, ready to go.
					self.model = tf.keras.Sequential(custom_architecture)

			'''NOTE NOTE NOTE END MODEL ARCHITECTURE DEVELOPMENT END NOTE END NOTE END NOTE'''

			#defining the lr reducer function for slowing or no improvement
			self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
				monitor	=	self.monitor_parameter,
				mode	=	self.monitor_condition,
				factor	=	rlr_factor, 
				patience=	int(rlr_patience), 
				min_lr	=	1e-6
			)

			#define an early stopping method for absolutely no improvement of the model in 2 * exceptionless-time
			self.train_stop = EarlyStopping(monitor=self.monitor_parameter, patience=2*rlr_patience, mode=self.monitor_condition, restore_best_weights=True)
			
			#here we are constructing the optimizer based off of specification
			#pulled the optimizers from tf/keras, adding anything should be relatively easy here
			match(optimizer_type):
				case 'SGD':
					self.optimizer = tf.keras.optimizers.SGD(**optimizer_kwarg)
				case 'Adam':
					self.optimizer = tf.keras.optimizers.Adam(**optimizer_kwarg)
				case 'AdamW':
					self.optimizer = tf.keras.optimizers.AdamW(**optimizer_kwarg)

			#compile constructed model with specified opt,loss,metrics
			self.model.compile(
				optimizer	=	self.optimizer
				,loss		=	self.loss_function
				,metrics	=	self.performance_metrics
			)

			#collect and save class weights
			self.class_weight=get_class_weights(y_train)

			#collecting and saving batch and epoch information
			self.batch_size = batch_size
			self.epochs		= epochs

			#declaring the LSTM property of the class and reformatting the data
			if(LSTM):
				self.LSTM	=	LSTM
				X_train, y_train= self.format_LSTM(X_train, y_train)
				X_test, y_test	= self.format_LSTM(X_test, y_test)

			#here we will fit the model and collect the training history data
			history = self.model.fit(X_train, y_train
				,epochs=self.epochs
				,shuffle=shuffle_train
				,verbose=train_verbose
				,validation_data=(X_test,y_test)
				,batch_size=self.batch_size
				,callbacks=[self.reduce_lr, self.train_stop]
				,class_weight=self.class_weight
			)




	def load_model(self, model):
		self.model = model

	def format_LSTM(X, y, time_steps):
		'''This function takes in a given X and y and returns an LSTM formatted version of the data'''

		X_lstm = []
	
		#for each sample
		for i in range(time_steps, len(X)):
			# Collect previous time_steps rows for X
			X_lstm.append(X[i-time_steps:i])  
			# The corresponding y value for the last time step in the sequence
	
		X_lstm = np.array(X_lstm)

		#naturally trim targets according to timestep length
		y_lstm = y[time_steps:]
		y_lstm = np.array(y_lstm)
	
		return X_lstm, y_lstm
	
	def save_as(self, name:str):
		'''This function will save a given model as a .keras / .xxxxx file'''
		#'model' here will have to be of type tf.keras.Sequential
		self.model.save(name)
