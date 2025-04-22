import numpy as np
import tensorflow as tf
from typing import Literal

def make_prll_trgt(data, index, offset):
	"""
	Shifts the nth column of a 2D numpy array by m values in the negative direction.

	Parameters:
		arr (numpy.ndarray): The input 2D array.
		n (int): The index of the column to shift.
		m (int): The number of positions to shift.

	Returns:
		numpy.ndarray: The shifted column as a new array.
	"""
	if not (0 <= index < data.shape[1]):
		raise ValueError("Column index out of range.")
	
	trgt = np.roll(data[:, index], -offset)
	return data[:-offset], trgt[:-offset]


def reformat_to_lstm(X, y, time_steps):
	X_lstm = []
	
	for i in range(time_steps, len(X)):
		# Collect previous time_steps rows for X
		X_lstm.append(X[i-time_steps:i])  
		# The corresponding y value for the last time step in the sequence
	
	X_lstm = np.array(X_lstm)

	y_lstm = y[time_steps:]
	y_lstm = np.array(y_lstm)
	
	return X_lstm, y_lstm

def make_vhs_target(data, close_index, s_index, mode:Literal['breakout','poshold']='breakout'):

	
	arr_close = data[:, close_index]
	arr_stoch   = data[:, s_index]

	length = len(arr_close)

	#list variables for holding metrics
	returns = np.zeros(length, dtype=np.float32)

	exit_disp = np.full(length, fill_value=-1, dtype=int)
	next_satisfied = -1

	#this loop is for collecting an array containing how long a trade would be held for at each instance of the backtesting data
	#forbidden trade areas hold zeros to denote zero holding time
	#hold_for (initial functionality of this file) sets a constant value for all instances of the dataset (ex; 15 mins holding time)
	#for i in range(length):


	#splitting exit type pre-sample-loop to minimize time-complexity
	match(mode):

		#this means if condition is satisfied, is forbidden area
		case 'breakout':

			#honestly used chatgpt for this idea, absolutely smartmode idea it came up with!!
			for i in reversed(range(1,len(arr_stoch))):

				#check to see if condition is satisfied
				if((arr_stoch[i]<80 and arr_stoch[i-1]>80) or (arr_stoch[i]>20 and arr_stoch[i-1]<20)):

					next_satisfied = i
						
				if(next_satisfied != -1):

					exit_disp[i] = next_satisfied - i

		#this means only forbidden area is the moment that condition is satisfied, 
		#if subsequently followed by condition satisfaction, it is neglected and is not forbidden 
		case 'poshold':

			#honestly used chatgpt for this idea, absolutely smartmode idea it came up with!!
			for i in reversed(range(1,len(arr_stoch))):

				#check to see if condition is satisfied
				if((arr_stoch[i]<80 and arr_stoch[i-1]>80) or (arr_stoch[i]>20 and arr_stoch[i-1]<20)):

					next_satisfied = i
						
				if(next_satisfied != -1):

					exit_disp[i] = next_satisfied - i

	#entire exit_disp array is completed, all match cases are exited
	#ensure any forbidden area flags (-1) are converted to zero
	#zero as an exit distance measurement is desireable as it will be thrown into calculation where:
	#we are taking natural log of exit price over entry price. if zero exit displacement then it is
	#some price n over some price n => equals 1 => ln(1) = 0, making zero return undesirable/unnoticable
	#now negating all significance of any trade here, making model gene pool ignore these areas! 
	#perfect woop woop woop woop woopen gangnem style
	exit_disp[exit_disp == -1] = 0 		

	#now we are going to loop through the array with the provided exit displacement variable array
	#and calculate returns from all bars under these circumstances
	for i in range(length):
			
		if(exit_disp[i] != 0):
			returns[i] = np.log(arr_close[i+exit_disp[i]]/arr_close[i])
			

	#metrics are built, return parallel metrics
	return returns


def make_day_dataset(X,
					 y,
					 scaler,
					 time_idx=1,
					 min_time=570,
					 max_time=900,
					 seq_len=15,
					 to_lstm=False,
					 include_days=None):
	"""
	Build a tf.data.Dataset that yields one day's windowed slice at a time.

	Args:
	  X: np.ndarray, shape (N, F)     — feature matrix
	  y: np.ndarray, shape (N,)       — parallel label array
	  time_idx: int                   — index of looping time feature (0–1200)
	  min_time, max_time: int         — inclusive window bounds
	  to_lstm: bool                   — if True, output has leading batch dim
	  include_days: list/np.ndarray   — which day IDs to include

	Returns:
	  tf.data.Dataset yielding (x_batch, y_batch):
		if to_lstm=False:
		  x_batch: (seq_len, F),   y_batch: (seq_len,)
		if to_lstm=True:
		  x_batch: (1, seq_len, F), y_batch: (1, seq_len)
	"""
	# infer day IDs by wrap‑around of the time feature
	times  = X[:, time_idx]
	day_id = np.concatenate([[0], np.cumsum(times[1:] < times[:-1])])

	#scaled time

	mintime = (min_time - scaler.mean_[1]) / scaler.scale_[1]
	maxtime = (max_time - scaler.mean_[1]) / scaler.scale_[1]

	# filter rows into your time window
	mask   = (times >= mintime) & (times <= maxtime)
	X_f    = X[mask]
	y_f    = y[mask]
	days_f = day_id[mask]

	# restrict to a subset of days if requested
	if include_days is not None:
		keep   = np.isin(days_f, include_days)
		X_f    = X_f[keep]
		y_f    = y_f[keep]
		days_f = days_f[keep]

	# generator: one day's slice per iteration
	def gen():
		for d in np.unique(days_f):
			x_batch = X_f[days_f == d].astype(np.float32)  # (var_len, F)
			y_batch = y_f[days_f == d].astype(np.float32)  # (var_len,)

			if seq_len is not None:
				cur_len = x_batch.shape[0]
				if cur_len >= seq_len:
					# truncate to first seq_len
					x_batch = x_batch[:seq_len]
					y_batch = y_batch[:seq_len]
				else:
					# pad at the front with zeros
					pad_amt = seq_len - cur_len
					pad_x = np.zeros((pad_amt, x_batch.shape[1]), dtype=np.float32)
					pad_y = np.zeros((pad_amt,), dtype=np.float32)
					x_batch = np.vstack([pad_x, x_batch])
					y_batch = np.concatenate([pad_y, y_batch])

			if to_lstm:
				x_batch = np.expand_dims(x_batch, 0)  # (1, seq_len, F)
				y_batch = np.expand_dims(y_batch, 0)  # (1, seq_len)

			yield x_batch, y_batch

	# declare the output shapes
	n_feats = X.shape[1]
	if to_lstm:
		spec = (
			tf.TensorSpec(shape=(1, None, n_feats), dtype=tf.float32),
			tf.TensorSpec(shape=(1, None)       , dtype=tf.float32),
		)
	else:
		spec = (
			tf.TensorSpec(shape=(None, n_feats), dtype=tf.float32),
			tf.TensorSpec(shape=(None,)        , dtype=tf.float32),
		)

	return tf.data.Dataset.from_generator(gen, output_signature=spec)