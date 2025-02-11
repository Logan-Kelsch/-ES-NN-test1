import random
import traceback
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import seaborn as sns
import matplotlib.pyplot as plt
import mplfinance as mpf
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
	naked_features=False
):
	'''This function must have X_raw come in with first features as high,low,close,volume,time,ToD,DoW'''

	#temporary assertion for limited implementation and mplfinance library knowledge, keeping feature plotting to max of 1
	assert (len(add_chart) < 5), \
		NotImplementedError(f"FATAL: Adding {len(add_chart)} charts is above the implemented maximum of 4.")

	#create batches based off of day loops
	batches = make_batches_with_ToD(X_raw, t_start, t_end)

	#iterable variable to keep batch looping parallel to prediction plotting
	sample_iter = 0

	#create batches for each day
	for batch in batches:

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
			if(predictions[sample_iter+sample]==1):
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

			add_plots = []#[mpf.make_addplot(features[1:i], panel=1, color='blue',secondary_y=False) for i in range(len(features[1:]))]
			for i, feat in enumerate(features.values):
				color = ((len(features.values) -i)/(len(features.values)+1))
				add_plots.append(mpf.make_addplot(feat[1:].clip(min=0,max=100), panel=1, color=(color/2,color/2,color), secondary_y=False))
			
			#naked features disabled allows for panel to have standard info lines such as 0,100 and 20,80
			if((naked_features==False) & (len(add_plots) > 0)):
				add_plots.append(mpf.make_addplot(pd.Series(0, index=range(len(df)-1)), panel=1, color='black', secondary_y=False))
				add_plots.append(mpf.make_addplot(pd.Series(100, index=range(len(df)-1)), panel=1, color='black', secondary_y=False))

			mpf.plot(df[1:], type='candle',style='yahoo',figratio=(16,9),figsize=(12,8), addplot=add_plots)
		
		#move sample prediction iterator up based off of size of this batch
		sample_iter+=len(batch[:])

def make_batches_with_ToD(X_raw, batch_begin, batch_end):
	batches = []
	current_batch = []

	for row in X_raw:
		if row[5] == batch_begin and current_batch:
			batches.append(np.array(current_batch))
			current_batch = []

		current_batch.append(row)
	
	if current_batch:
		batches.append(np.array(current_batch))

	return batches

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