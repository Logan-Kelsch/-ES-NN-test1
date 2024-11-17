'''
    Logan Kelsch
    
    This file will be used to hold all of the
    data preprocessing / processing / prep functions
'''

import numpy as np
import pandas as pd

#setting data for LSTM
def reformat_to_lstm(X, time_steps=5):
    X_lstm, y_lstm = [], []
    
    for i in range(time_steps, len(X)):
        # Collect previous time_steps rows for X
        X_lstm.append(X[i-time_steps:i])  
        # The corresponding y value for the last time step in the sequence
    
    X_lstm = np.array(X_lstm)
    
    return X_lstm

'''
    ATTENTION!!!
        FOR THE NEXT TWO FUNCTIONS, THEY WILL ONLY OPERATE CORRECTLY
        GIVEN THAT THE TWO TIME FEATURES ARE REMOVED PRE-PCA AND
        REAPPENDED TO THE FUNCTION PROVIDED SET BEFORE THEY ARE CALLED.
'''

def remove_zero_mo_samples(X, y, Xfeatures, timeSteps=5):
    # Get the 'MO' column (index 34 for 0-based indexing) for all time steps and samples
    non_zero_indices = (X[:, timeSteps-1, X.shape[2]-1] >= 0)
    # Filter X and y using these indices
    X_filtered = X[non_zero_indices]
    y_filtered = y[non_zero_indices]
    print(f'Remaining Sample Count - remove_zero_mo_samples:\n\t{len(X_filtered)}')
    return X_filtered, y_filtered

def remove_extra_filter(X, y, Xfeatures, timeSteps, timeStart, timeStop):
    indices = (X[:, timeSteps-1, X.shape[2]-3] >= timeStart)#-3 is ToD, this value is 9:30am
    X = X[indices]
    y = y[indices]
    indices = (X[:, timeSteps-1, X.shape[2]-3] <= timeStop)#-3 is ToD, this value is 12:00pm
    X = X[indices]
    y = y[indices]
    print(f'Remaining Sample Count - remove_extra_filter:\n\t{len(X)}')
    return X, y

def target_setter(data, testFor):
    match testFor:
        case 'r1':
            data = data.drop(columns=['r2','r3','r5','r10','r15','r30','r60'])
        case 'r2':
            data = data.drop(columns=['r1','r3','r5','r10','r15','r30','r60'])
        case 'r3':
            data = data.drop(columns=['r1','r2','r5','r10','r15','r30','r60'])
        case 'r5':
            data = data.drop(columns=['r1','r2','r3','r10','r15','r30','r60'])
        case 'r10':
            data = data.drop(columns=['r1','r2','r3','r5','r15','r30','r60'])
        case 'r15':
            data = data.drop(columns=['r1','r2','r3','r5','r10','r30','r60'])
        case 'r30':
            data = data.drop(columns=['r1','r2','r3','r5','r10','r15','r60'])
        case 'r60':
            data = data.drop(columns=['r1','r2','r3','r5','r10','r15','r30'])
    return data