'''
	NOTE NOTE NOTE 
	This file will contain all of the custom neural network code for any given implementation.
	Going to try to hopefully actually allow for all LSTM to be done within this file???
	If so that would allow for much more expandability with simple implementation of lstm testing and training,
	especially if we can go back and make sure all custom train/test functions are passed split data,
	making anything where LSTM is built here non-overlapping no matter what.

	NOTE delete this once this is considered and fulfilled
'''

import _Utility
from importlib import reload
reload(_Utility)
import tensorflow as tf
import numpy as np
import joblib
from keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
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
		self._class_weight	=	None
		self._epochs		=	None
		self._batch_size	=	None
		self._LSTM			=	None
		
		#prediction mode specific characteristics
		match(predict_mode):
			case 'Classification':
				self._target_activation	=	'sigmoid'
				self._target_neurons		=	1
				self._performance_metrics=	['precision','recall','accuracy']
				self._monitor_parameter	=	'val_accuracy'
				self._monitor_condition	=	'max'
				self._loss_function		=	'binary_crossentropy'
			case 'Multi-Class':
				self._target_activation	=	'softmax'
				self._target_neurons		=	class_count
				self._performance_metrics=	['precision','recall','accuracy']
				self._monitor_parameter	=	'val_accuracy'
				self._monitor_condition	=	'max'
				self._loss_function		=	'categorical_crossentropy'
			case 'Regression':
				self._target_activation	=	'linear'
				self._target_neurons		=	1
				self._performance_metrics=	['R2Score','root_mean_squared_error']
				self._performance_metrics=	'val_mse'
				self._monitor_condition	=	'min'
				self._loss_function		=	'mse'

		#Learning rate reduction function 
		self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
			monitor=self._monitor_parameter,
			mode=self._monitor_condition,
			factor=0.75, 
			patience=1000, 
			min_lr=1e-6
		)
		
		
	
	#Here are a couple property functions, will probably miss a few and will need to reup this frequently

	@property
	def target_activation(self):
		return self._target_activation

	@property
	def target_neurons(self):
		return self._target_neurons
	
	@property
	def performance_metrics(self):
		return self._performance_metrics

	@property
	def monitor_parameter(self):
		return self._monitor_parameter

	@property
	def monitor_condition(self):
		return self._monitor_condition

	@property
	def loss_function(self):
		return self._loss_function
	
	'''This set of 3 characteristics may be buggy, before testing, unsure of ease of accessing these variables'''
	
	@property
	def reduce_lr_factor(self):
		return self._reduce_lr.factor
	
	@property
	def reduce_lr_patience(self):
		return self._reduce_lr.patience
	
	@property
	def reduce_lr_min_lr(self):
		return self._reduce_lr.min_lr
	
	#Here are the setter functions twinning the property functions, also will need reup with updates

	@target_activation.setter
	def target_activation(self, new:str):
		self._target_activation = new

	@target_neurons.setter
	def target_neurons(self, new:int):
		self._target_neurons = new
	
	@performance_metrics.setter
	def performance_metrics(self, new:list):
		self._performance_metrics = new

	@monitor_parameter.setter
	def monitor_parameter(self, new:str):
		self._monitor_parameter = new

	@monitor_condition.setter
	def monitor_condition(self, new:str):
		self._monitor_condition = new

	@loss_function.setter
	def loss_function(self, new:str):
		self._loss_function = new

	'''This set of 3 characteristics may be buggy, before testing, unsure of ease of accessing these variables'''
	
	@property
	def reduce_lr_factor(self):
		return self._reduce_lr.factor
	
	@property
	def reduce_lr_patience(self):
		return self._reduce_lr.patience
	
	@property
	def reduce_lr_min_lr(self):
		return self._reduce_lr.min_lr
	
	#some more properties, no setter function because they are overwritten in build_fit
	@property
	def epochs(self):
		return self._epochs
	
	@epochs.setter
	def epochs(self, new:int):
		self._epochs = new
	
	@property
	def batch_size(self):
		return self._batch_size
	
	@batch_size.setter
	def batch_size(self, new:int):
		self._batch_size = new
	
	@property
	def LSTM(self):
		return self._LSTM
	
	@LSTM.setter
	def LSTM(self, new:bool):
		self._LSTM = new
	
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

	def build(
		self,
		X_train
		,y_train
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
		,external_cw	:	dict												=	None 
		,custom_val_data:	tuple												=	None
	):
		'''This function builds the model in NN.model, calls .compile and .fit and returns the training history'''

		#this function should create and build self.model (of type tf.keras.Sequential)
		#then it will call self.model.compile
		#then it will call self.model.fit
		#and it will then return history

		
		#collect and save class weights
		if(external_cw == None):
			#no external class weights are entered
			self._class_weight=get_class_weights(y_train)
		else:
			#external class weights are entered
			self._class_weight=external_cw

		#simple method of ensuring and printing which device will be used for computation
		cmp = 'C'
		if(tf.config.list_physical_devices('GPU')):
			cmp = 'G'
		with tf.device('/'+cmp+'PU:0'):
			print('Running on: '+cmp+'PU\n')

			'''NOTE NOTE NOTE BEGIN ARCHITECTURE DEVELOPMENT END NOTE END NOTE END NOTE'''

			match(architecture):
				case 'default_shallow':
					
					#adding a single hidden layer of neuron size -> 
					# 							lowest power of 2 >= size of featurespace coming in
					if(LSTM):
						seq = [tf.keras.layers.Input(shape=(time_steps,int(X_train.shape[1]))), tf.keras.layers.LSTM(int(2**np.ceil(np.log2(X_train.shape[1]))), return_sequences=False)]
					else:
						seq = [tf.keras.layers.Input(shape=(int(X_train.shape[1]),)), tf.keras.layers.Dense(int(2**np.ceil(np.log2(X_train.shape[1]))), activation='relu')]
					#adding the output layer
					seq.append(tf.keras.layers.Dense(self._target_neurons, self._target_activation))
					#combine in to actual keras interpretable information
					self.model = tf.keras.Sequential(seq)
				case 'default_deep':
					
					#initial depth of model, neuron length of first hidden layer, and 2 * depth_init is non-input neuron volume of model
					depth_init = int(np.ceil(np.log2(X_train.shape[1])))

					#The default deep model will be of an exponential-tree shape, 
					#where hidden layer 1 is of size 1.5 * 2^ceiling(logbase2(# features)) - 3
					#adding a single hidden layer of neuron size -> 
					# 							lowest power of 2 >= size of featurespace coming in
					if(LSTM):
						seq = [tf.keras.layers.Input(shape=(time_steps,int(X_train.shape[1]))), tf.keras.layers.LSTM(int(2**np.ceil(np.log2(X_train.shape[1]))), return_sequences=False)]
					else:
						seq = [tf.keras.layers.Input(shape=(int(X_train.shape[1]),)), tf.keras.layers.Dense(int(2**np.ceil(np.log2(X_train.shape[1]))), activation='relu')]
					
					#appending connective section between first large layer and rest of hidden section
					seq.append(tf.keras.layers.Dropout(0.25))
					seq.append(tf.keras.layers.BatchNormalization())
					
					#add remaining layers here, skipping every other power, found it near redundant in my experience
					for layer_size in range(depth_init-1, self._target_neurons+1, -1):
						seq.append(tf.keras.layers.Dense(2**layer_size, activation='relu'))
						seq.append(tf.keras.layers.Dropout(0.25))
						seq.append(tf.keras.layers.BatchNormalization())

					#adding the output layer
					seq.append(tf.keras.layers.Dense(self._target_neurons, self._target_activation))

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
			self._train_stop = EarlyStopping(monitor=self._monitor_parameter, patience=2*rlr_patience, mode=self._monitor_condition, restore_best_weights=True)
			
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
				,loss		=	self._loss_function
				,metrics	=	self._performance_metrics
			)

			#collecting and saving batch and epoch information
			self._batch_size = batch_size
			self._epochs		= epochs
			self._shuffle_train = shuffle_train
			self._train_verbose = train_verbose
			self._custom_val_data=custom_val_data
			self._LSTM = LSTM
			self._time_steps = time_steps


	def fit(self, X_train, y_train, X_test, y_test, custom_val_data=None):

		#quick delcaration to ensure proper validaiton data going in 
		if(self._custom_val_data == None):
			#no custom validation data was entered, input default
			self._custom_val_data = (X_test, y_test)
		else:
			#This suggests that custom validation data was input
			pass#thus the variable can be left alone
		
		#simple method of ensuring and printing which device will be used for computation
		cmp = 'C'
		if(tf.config.list_physical_devices('GPU')):
			cmp = 'G'
		with tf.device('/'+cmp+'PU:0'):
			print('Running on: '+cmp+'PU\n')

			#declaring the LSTM property of the class and reformatting the data
			if(self._LSTM):
				X_train, y_train= self.format_LSTM(X_train, y_train, self._time_steps)
				X_test, y_test	= self.format_LSTM(X_test, y_test, self._time_steps)

			#quick delcaration to ensure proper validaiton data going in 
			if(custom_val_data == None):
				#no custom validation data was entered, input default
				self._custom_val_data = (X_test, y_test)
			else:
				#This suggests that custom validation data was input
				pass#thus the variable can be left alone

			#here we will fit the model and collect the training history data
			history = self.model.fit(X_train, y_train
				,epochs=self._epochs
				,shuffle=self._shuffle_train
				,verbose=self._train_verbose
				,validation_data=self._custom_val_data
				,batch_size=self._batch_size
				,callbacks=[self.reduce_lr, self._train_stop]
				,class_weight=self._class_weight
			)
			_Utility.graph_loss(self._epochs, history)

	def predict(self
			,X
			,y								=	None
			,threshold			:	float	=	0.5
			,use_class_weight	:	bool	=	False
	):
		if(use_class_weight):
			#set class weights
			pass

		#if(self._LSTM):
		#	X, y = self.format_LSTM(X, y, self._time_steps)
		y_pred = self.model.predict(X)
		y_pred = (y_pred > threshold)

		#if(self._LSTM):
		#	return y_pred, y
		return y_pred


	def load_model(self, model):
		self.model = model

	def format_LSTM(self, X, y, time_steps=None):
		'''This function takes in a given X and y and returns an LSTM formatted version of the data'''

		if(time_steps == None):
			time_steps = self._time_steps

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
		self.model.save(name+'.keras')
		#self.model = None
		#self.optimizer = None

	def dump(self):
		return self

	def load_ext(self, name:str):
		'''This function is to load in the model from file'''

		self.model = load_model(name)