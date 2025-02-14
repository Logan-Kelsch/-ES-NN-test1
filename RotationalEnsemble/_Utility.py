import random
import traceback
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import Literal
import pandas as pd
import copy

#function to simplify code visualization/readability (primary for verbose printouts)
def do_nothing():
	pass

class model_params:
	def __init__(self):
		self.model_type = None
		self.target_activation = None
		self.target_neurons = None
		self.performance_metrics = None
		self.loss_function = None
		self.target_time = None
		self.monitor_parameter = None
		self.monitor_condition = None
		#classification specfic
		self.class_split_val = None
		self.num_classes = None
		self.class_weights = None
		
''' NOTE
	ANY CHANGE OF METRICS ARRAY WILL CAUSE ERROR IN 
	performance_printout.py. Either implement array methodology
	in printout of history in performance_printout file, or 
	manually change function in performance_prinout to match
	metrics used.
	NOTE
	Did not implement array methodology as I don't immediately
	foresee changing the metrics used..
'''
#this function initiates a parameters class for implementation
#of fast changable variables for all variables that differ in
#           REGRESSION, BINARY, AND MULTICLASS CLASSIFICATION
def get_model_params(m_type, target_time, c_split_val, c_class_cnt):
	#initialize class and pull in model type
	params = model_params()
	params.model_type = m_type
	params.target_time = target_time
	match(params.model_type):
		case 'Regression':#################################
			params.target_activation = 'linear'
			params.target_neurons = 1
			params.performance_metrics = \
				['R2Score','root_mean_squared_error']
			params.loss_function = 'loss'
			params.monitor_parameter = 'val_loss'
			params.monitor_condition = 'min'
		case 'Classification':#############################
			params.num_classes = c_class_cnt
			params.class_split_val = c_split_val
			params.target_activation = 'sigmoid' if \
				(c_class_cnt == 2) else 'softmax'
			params.loss_function = \
				'binary_crossentropy' if (c_class_cnt == 2)\
				else 'categorical_crossentropy'
			params.target_neurons = 1 if (c_class_cnt == 2)\
				else c_class_cnt
			params.performance_metrics = \
				['precision','recall','accuracy']
			params.monitor_parameter = 'val_accuracy'
			params.monitor_condition = 'max'
		case _:
			raise ValueError(f"Invalid m_type (model type) \
							 {params.model_type}.")
	return params

def pick_n_from_list(
    n		:	int
	,nums	:	list
)	->	list:
    '''Function picks n number from the inclusive range of [start, end]
    Params:
    - n
    -	_Number of integers picked from the range
    - nums
    -	_list of numbers to be picked from
    '''
    #ensure that this function isn't computed with impossible inputs
    if(n > len(nums)):
        print(f'FATAL: Tried to pick {n} unique numbers from list of size {len(nums)}.')
        traceback.print_exc()
        raise
    
    #range initiation
    random.shuffle(nums)
    picks = []
    
    #pick the dang numbers WITHOUT the same dang numbers
    for i in range(n):
        picks.append(nums.pop())
        
    #GIVE ME BACK MY DANG NUMBERS THAT ARE UNIQUE AND N LENGTH OF RANGE [START,END]
    return picks

#this function will generate class weights if the model is classification
def get_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {int(cls): weight for cls, weight in zip(classes,weights)}
    return class_weights

#This function works as a model that only predicts the provided value for every instance.
def dummy_predict(X_test, prediction=1):
	#create the prediction array for return
	y_pred = np.array([])

	#go through each sample of the X_test set
	for sample in X_test:
		y_pred = np.append(y_pred, prediction)

	return y_pred

def plot_standard_line(
		y
		,X=None
		,xlabel=None
		,ylabel=None
		,legend=None
		,axhline=None
		,axvline=None
		,label:str='Line Plot'):
	if(X == None):
		X = range(len(y))
	plt.figure(figsize=(12, 6))
	plt.plot(X, y)
	if(axhline != None):
		if(len(axhline)>1):
			for value in axhline:
				plt.axhline(value)
	if(axvline != None):
		if(len(axvline>1)>1):
			for value in axvline:
				plt.axvline(value)
	plt.show()

#function to print out loss function, should be universal
def graph_loss(epochs, history):
    plt.figure(figsize=(6, 3))
    plt.plot(epochs[1:], history.history['loss'][1:], 'black', label='Training Loss')
    plt.plot(epochs[1:], history.history['val_loss'][1:], 'red', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

#function takes the percent difference that val1 is of val2 as full values (55.5%, not .555)
def po(val1, val2, as_decimal=False, round_to=8):
	'''po is shorthand for 'percent of', for quicky repetitive typing in other files'''
	m = 1 if as_decimal else 100
	return round((m * val1 / val2) , round_to)

def function_executor(func, args):
	'''This function takes in a function and a tuple of arguments, no use of keywords here though.'''
	return func(*args)

def swap(val1, val2):
	'''simple swapping function, nothing special'''
	return val2, val1

def get_cm_values(y_true, y_pred):
	'''This function takes in the true and prediction values of a given model and set, and returns values from 0-3 based on its type.
	FORMAT:
	-	0	1
	-	2	3	
	'''
	cm_vals = []
	for i in range(len(y_pred)):
		if(y_true[i] == 0):
			if(y_pred[i] == 0):
				cm_vals.append(0)
			if(y_pred[i] == 1):
				cm_vals.append(1)
		if(y_true[i] == 1):
			if(y_pred[i] == 0):
				cm_vals.append(2)
			if(y_pred[i] == 1):
				cm_vals.append(3)
	return cm_vals

def show_confusion_matrix(
		y_true,
		y_pred
		,figsize	=	(8,6)
		,cmap		=	'Greens'
		,xticklabels=	range(2)
		,yticklabels=	range(2)
		,xlabel		=	'Predicted'
		,ylabel		=	'True'
		,title		=	'Confusion Matrix'
):
	#Create the confusion matrix
	cm = confusion_matrix(y_true, y_pred)
	# Plot the confusion matrix using seaborn
	plt.figure(figsize=figsize)
	sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, \
				xticklabels=xticklabels, yticklabels=yticklabels)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def get_precision(y_true, y_pred):
	return f'{round(precision_score(y_true, y_pred)*100, 2)}%'

def get_accuracy(y_true, y_pred):
	return f'{round(accuracy_score(y_true, y_pred)*100, 2)}%'

def graph_range(function, kw, kw_range, show_graph:bool=True, **kwargs):
	values = []
	for i in kw_range:
		local_kwargs = kwargs.copy()
		local_kwargs[kw] = i
		values.append(function(**local_kwargs))
	
	if(show_graph):
		plt.figure(figsize=(8,4))
		plt.plot(values, kw_range, 'black')
		plt.show

	return values

def show_predictions_chart(
	X_raw, 
	predictions, 
	t_start=645,
	t_end=800, 
	add_chart:list=[], 
	fss=None, 
	naked_features=False,
	signal: any = None
):
	'''This function must have X_raw come in with first features as high,low,close,volume,time,ToD,DoW'''

	#temporary assertion for limited implementation and mplfinance library knowledge, keeping feature plotting to max of 1
	assert (len(add_chart) < 5), \
		NotImplementedError(f"FATAL: Adding {len(add_chart)} charts is above the implemented maximum of 4.")

	#create batches based off of day loops
	batches, batch_root_indices = make_batches_with_ToD(X_raw, t_start, t_end)

	#iterable variable to keep batch looping parallel to prediction plotting
	sample_iter = 0

	#signals variable to append each signal going off
	signals = []

	#create batches for each day
	for batch_index, batch in enumerate(batches):

		#collect the high low and close and generate open from the given raw dataset
		h = batch[:,0]
		l = batch[:,1]
		c = batch[:,2]
		o = np.roll(c, shift=1)

		#check if plotting a parallel feature was requested
		if(len(add_chart)!=0):
			ft = []

			#now append each feature column into a list
			for feature_index in add_chart:
				ft.append(batch[:,feature_index])

			#also check if a collection of feature subsets were brought into
			#the function for printout confirmation of which feature is being printed
			if(fss != None):
				print(f"Plotting Features: {[get_name_from_fss(fss, i) for i in add_chart]}")

		#small for loop to force direction of candle based on prediction of model
		for sample in range(len(c)):
			#if predicts 1
			if(predictions[batch_root_indices[batch_index]+sample]==1):
					if(c[sample]<o[sample]):#force green
						c[sample],o[sample] = o[sample],c[sample]
			else:#if predicts 0
				if(c[sample]>o[sample]):#force red
					c[sample],o[sample] = o[sample],c[sample]


		data = {
			'Date':range(0,len(batch[:])*1000000000,1000000000),
			'Open':o,
			'High':h,
			'Low':l,
			'Close':c
		}
		df = pd.DataFrame(data)
		df['Date'] = pd.to_datetime(df['Date'])
		df.set_index('Date',inplace=True)

		#check whether or not chart printout should be with a feature
		if(len(add_chart) == 0):
			#this is reached if no feature is being shown parallel to the given candles
			mpf.plot(df[1:], type='candle',style='yahoo',figratio=(20,8))

		else:
			#this is reached if a requested feature is being printed parallel to the candle charts
			features = pd.DataFrame(ft)

			add_plots = []
			for i, feat in enumerate(features.values):
				color = ((len(features.values) -i)/(len(features.values)+1))
				add_plots.append(mpf.make_addplot(feat.clip(min=0,max=100), panel=1, color=(color/2,color/2,color), secondary_y=False))

			#this area is for if singals are included
			#excluding the use of signals if no plots have been added
			if(len(add_plots)>0 and signal!=None):
				
				#checking to see if the given signal is a number, if so, goes into this if statement
				if(type(signal)==int or type(signal)==float):

					#checking to see if the value is around the overbought area
					#then would use this as a greater than n parameter
					if(signal >= 75 & signal <= 100):
						
						'''NOTE NOTE GOING TO MAKE SIGNALS TEMPORARILY ONLY APPLY TO THE FIRST FEATURE PLOTTED END#NOTE END#NOTE'''
						#this feature contains NAN values
						#also differentiating between whether there is more than one feature to grab signal from 
						if(len(add_plots)>1 or signal==None):

							#more than one feature being plotted
							raise KeyError('signal not supported with multiple plots')
							#signal_feature = pd.Series(np.where(features.iloc[1:,0].values >= signal, features.iloc[1:,0].values, np.nan))
						else:
							
							#only one feature being plotted
							#this one goes ontop of the feature plot
							signal_feature = batch[:,add_chart[0]]
							#this one goes ontop of the candle plot
							#the purpose of this one is to show the signal overlap between model AND feature
							signal_candle  = copy.deepcopy(signal_feature)

							color = np.array([])

							#checking each sample for whether there is a signal
							for i in range(len(signal_feature)):

								#checking for feature signal failure
								if(signal_feature[i] <= signal):

									#if no signal, nullify this sample value in this vector
									signal_feature[i] = np.nan
									signal_candle[i]  = np.nan
									color = np.append(color, 'red')

								#feature signal success case
								else:

									#signal is in lower half of range
									if(signal_feature[i] <= ((100+signal)/2)):
										color = np.append(color, 'blue')
									else:
										#purple temporarily denoting stronger signal
										color = np.append(color, 'purple')

									#model signal success case
									if(df['Close'].iloc[i] > df['Open'].iloc[i]):

										#append this location to list of signals
										#expln: batch specific offset + sample-of-batch specific offset + single sample visualization trimming offset
										signals.append(batch_root_indices[batch_index] + i)

										#set plottable value at low of candle, +1 is accounting for [1:] trimming in definition of signal_feature
										signal_candle[i] = df['Low'].iloc[i]

									#model signal failure case
									else:
										#if no double signal, nullify this sample value in this vector
										signal_candle[i] = np.nan
							
							
						#append the candle specific plot of the signal
						add_plots.append(mpf.make_addplot(signal_candle, type='scatter',markersize=15,marker='^',color=color,secondary_y=False))
						#append the feature specific plot of the signal
						add_plots.append(mpf.make_addplot(signal_feature, type='scatter',markersize=5,color='black',secondary_y=False,panel=1))

					else:
						raise NotImplementedError(f'Signal desired has not yet been implemented, but is recognized as a number: {signal}')
			
			elif(signal != None):
				raise NotImplemented(f'Signal desired has not yet been implemented, and is not recognized as a number, but as type: {type(signal)}')

			else:
				#this case is reached when no signal is desired
				pass

			
			#naked features disabled allows for panel to have standard info lines such as 0,100 and 20,80
			if((naked_features==False) & (len(add_plots) > 0)):
				add_plots.append(mpf.make_addplot(pd.Series(0, index=range(len(df))), panel=1, color='black', secondary_y=False))
				add_plots.append(mpf.make_addplot(pd.Series(100, index=range(len(df))), panel=1, color='black', secondary_y=False))

			mpf.plot(df[:], type='candle',style='yahoo',figratio=(16,9),figsize=(12,8), addplot=add_plots)
		
		#move sample prediction iterator up based off of size of this batch
		sample_iter+=len(batch[:])


	#this function will return the X indices of where desired signals go off
	return signals



def make_batches_with_ToD(X_raw, batch_begin, batch_end):
	'''This function returns the batches of samples as well as a list of parallel length containing sample indices of first sample of each batch'''
	batches = []
	current_batch = []

	batch_root_indices = []

	for row_index, row in enumerate(X_raw):

		if(row[5]==batch_begin):

			#toggle on switch to collect samples
			is_collecting = True

			#reset current working batch
			current_batch = []

			#collect first sample as the root index of the batch
			batch_root_indices.append(row_index)

		if(row[5]==batch_end):
			
			#turn off switch to collect samples
			is_collecting = False

			#append batch to grouping if a batch was made
			if(current_batch):
				current_batch.append(row)
				batches.append(np.array(current_batch))

		#now collect each sample in switch collecting toggle switch is ON
		if(is_collecting):
			current_batch.append(row)
	
	#this case is reached if the batch_end value was never reached, and a 
	#collection was building (Ex: dataset ends mid time range to plot)
	if(current_batch):
		batches.append(np.array(current_batch))

	for i in range(len(batches)):
		print(len(batches[i]),batch_root_indices[i])

	return batches, batch_root_indices

def get_name_from_fss(fss:list=[], index:int=-1):
	'''This function takes a given feature subsets list and returns the feature name from a requested index value'''

	#check each feature subset (dict) within the list of feature subsets
	for subset in fss:
		#if the index value exists within the feature subset
		if(index in subset):
			#return the value to the given key
			return subset[index]
	
	#getting here means all feature subsets were checked and feature index is not contained within this
	return None

def backtester(
	X_raw, 
	signals,
	method:Literal['time_held','time_held_range'] = 'time_held',
	value:any=None):
	'''This function is going to analyze the performance of the signals collected using various tactics'''

	#approaching backtesting method based on method desired
	match(method):

		case 'time_held':
			
			#enforcing proper variable types
			assert (type(value)==int or type(value)==float),\
				f'Type of parameter value must be int or float for method time_held, got value type {type(value)}.'
			


		case 'time_held_range':
			pass