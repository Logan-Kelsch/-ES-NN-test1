import random
import traceback
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

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