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



#This class is for a traditional neural network, using tensorflow
class NN:
	def __init__(
		self
		,pred_mode	=	'Classification'
	):
		self.pred_mode	=	pred_mode
