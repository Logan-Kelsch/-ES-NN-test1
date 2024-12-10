'''
    Logan Kelsch
    
    This file will be used to hold all of the
    data preprocessing / processing / prep functions
'''

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from feature_creation import *

#class that holds all interchangable variables used in 
#model building between regression and classification
#                              (binary and multiclass)
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
            params.loss_function = 'mse'
            params.monitor_parameter = 'val_mse'
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

#function will return dataset after dropping all unused targets
def set_target(data, params):
    if(params.model_type == 'Regression'):
        data = data.drop(columns=tn_classification())
        data = data.drop(columns=tn_regression_excpetion(params.target_time))
    else:   ### CLASSIFICATION #################
        data = data.drop(columns=tn_regression())
        data = data.drop(columns=tn_classification_exception(\
            params.num_classes, params.class_split_val, params.target_time))
    print(f'TARGET: {data.columns[-1]}')
    return data

#function that uses the model.predict function, but allows
#for the execution of regression and classification predictions
#while the proper and required data structures hold true
def model_predict(model, params, X_eval, y_eval):
    if(params.model_type == 'Regression'):
            #regression scenario
        y_pred = model.predict(X_eval)
    else:   #binary class scenario
        if(params.num_classes == 2):
            y_pred = model.predict(X_eval)
            y_pred = (y_pred > 0.5)
        else: #multiclass scenario
            #convert one-hot to class indices if needed
            y_eval = np.argmax(y_eval, axis=1)
            y_pred = np.argmax(model.predict(X_eval), axis=1)
    return y_pred, y_eval

#this function is used after X y split for if y needs
#binarized for classification
def y_preprocess(params, y):
    if(params.model_type == 'Classification'):
        #Encoding data
        labelencoder = LabelBinarizer()
        y = labelencoder.fit_transform(y)
    return y, labelencoder

from sklearn.utils.class_weight import compute_class_weight

#this function will generate class weights if the model is classification
def update_class_weights(y, params):
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = {int(cls): weight for cls, weight in zip(classes,weights)}
    params.class_weights = class_weights
    return params

#setting data for LSTM
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

'''
    ATTENTION!!!
        FOR THE NEXT TWO FUNCTIONS, THEY WILL ONLY OPERATE CORRECTLY
        GIVEN THAT THE TWO TIME FEATURES ARE REMOVED PRE-PCA AND
        REAPPENDED TO THE FUNCTION PROVIDED SET BEFORE THEY ARE CALLED.
'''

def remove_extra_filter(X, y, timeSteps, timeStart, timeStop):
    indices = (X[:, timeSteps-1, X.shape[2]-3] >= timeStart)#-3 is ToD, this value is 9:30am
    X = X[indices]
    y = y[indices]
    indices = (X[:, timeSteps-1, X.shape[2]-3] <= timeStop)#-3 is ToD, this value is 12:00pm
    X = X[indices]
    y = y[indices]
    print(f'Remaining Sample Count - remove_extra_filter:\n\t{len(X)}')
    return X, y

#returns indices of sample times that are KEPT, to filter out all others
#after LSTM data formation
def grab_wanted_times(X, start_time, end_time, time_steps):
    #[:, 5] is time of day in minutes

    indices = np.where((X[:-time_steps, 5] >= start_time) & (X[:-time_steps, 5] <= end_time))[0]
    '''
    head_indices = (X[:, 5] >= start_time)
    tail_indices = (X[:, 5] <= end_time)

    #intersection of both
    indices = np.intersect1d(head_indices, tail_indices, return_indices=True)
    '''

    return indices

#this function is going to replace the model.fit function
#this is to allow for the use of class weights.
#this function also returns the history
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

#this function may be considered redundant
#take indices of samples to be kept and remove all others
def filter_times(X, y, keep_indices):
    #keep_indices = keep_indices[:]
    return X[keep_indices,:], y[keep_indices]

def normalize_from_tt_split(X_sctran, X_scfit, test_size):
    fit_cutter = int(len(X_scfit)*(1-test_size))
    #scale all data to that of the projected training set datapoints
    X_fit = X_scfit[:fit_cutter]

    scaler1 = StandardScaler()
    scaler2 = RobustScaler()
    scaler3 = MinMaxScaler(feature_range=(0,1))

    scaler = scaler1
    scaler.fit(X_fit)
    return scaler.transform(X_sctran)

#this function will need to be re-evaluated if 
#various test_sizes are used around different models
def reform_with_PCA_isolated(X_pcatran, X_pcafit, test_size, num_isol_feats, comps_PCA):
    fit_cutter_fit = int(len(X_pcafit)*(1-test_size))
    X_main_features_fit = X_pcafit#[:, :-num_isol_feats]
    X_fit = X_main_features_fit[:fit_cutter_fit]#update fit values for PCA

    X_main_features = X_pcatran#[:, :-num_isol_feats]
    #X_time_features = X_pcatran[:, -num_isol_feats:]
    
    pca = PCA()

    pca = PCA(n_components = comps_PCA, random_state=0)
    pca.fit(X_fit)
    X_main_features = pca.transform(X_main_features)

    #return np.hstack((X_main_features, X_time_features))
    return X_main_features

#this function cuts a few corners and saves some lines in data processing phase.
#This function takes the set of all samples and applies the sci-kit learn
#train test split function to split the 3D LSTM data into 3 sections
#'_train' is for model training
#'_val' is for the validation set
#'_ind' is for the independent set
def split_into_train_val_ind(X, y, test_size, indp_size, time_steps):
    #split data into trained and not trained
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, train_size=(1-test_size-indp_size), shuffle=False)
    #calcluate precent of test samples that are for validaiton set
    val_size = (test_size)/(test_size+indp_size)
    #split test samples into what is used in validation and what is for
    #post-model building performance testing
    X_val, X_ind, y_val, y_ind =\
        train_test_split(X_test, y_test, train_size=val_size, shuffle=False)
    #trim out all overlapping samples in the time step dimension of LSTM layers
    #the front half of training samples are not overlapping and need no trimming
    X_val = X_val[time_steps:]
    y_val = y_val[time_steps:]
    X_ind = X_ind[time_steps:]
    y_ind = y_ind[time_steps:]
    return X_train, X_val, X_ind, y_train, y_val, y_ind

#this function has the best name of all time
#this function is used when the standard scaler is used, and removes all samples after
#   LSTM data structuring that are outside of a given deviation range?
def remove_stinky_samples():
    #get stinky samples indices
    #filter out sinky samples
    #return good smelling samples
    return


#LOSS FUNCTION
from keras.saving import get_custom_objects
from keras.saving import register_keras_serializable
get_custom_objects().clear()
#CUSTOM LOSS 1_______________________________________________________________________________________________
from keras.src import ops
from keras.src.losses.loss import squeeze_or_expand_to_same_rank
@register_keras_serializable(name="skew_loss")
def skew_loss(y_true,y_pred,sFact=4):
    fact = 20
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = ops.convert_to_tensor(y_true, dtype=y_pred.dtype)
    #y_true, y_pred = squeeze_or_expand_to_same_rank(y_true, y_pred)
    error = ops.subtract(y_pred, y_true)
    a = ops.convert_to_tensor(ops.cast(abs(y_pred-y_true)*fact <= 1,tf.float32), dtype=tf.float32)
    b = ops.convert_to_tensor(ops.cast(y_pred < 0,tf.float32), dtype=tf.float32)
    h = ops.convert_to_tensor(2, dtype=error.dtype)
    return ops.mean(
        ops.where(
            a==1,
            0,
            ops.square(error)-(1/fact)
        ))
