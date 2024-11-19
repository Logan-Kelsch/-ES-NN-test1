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
#import matplotlib as plt

#setting data for LSTM
def reformat_to_lstm(X, time_steps):
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

def remove_zero_mo_samples(X, y, timeSteps):
    # Get the 'MO' column (index 34 for 0-based indexing) for all time steps and samples
    non_zero_indices = (X[:, timeSteps-1, X.shape[2]-1] >= 0)
    # Filter X and y using these indices
    X_filtered = X[non_zero_indices]
    y_filtered = y[non_zero_indices]
    print(f'Remaining Sample Count - remove_zero_mo_samples:\n\t{len(X_filtered)}')
    return X_filtered, y_filtered

def remove_extra_filter(X, y, timeSteps, timeStart, timeStop):
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


def normalize_from_tt_split(X_sctran, X_scfit, test_size):
    fit_cutter = int(len(X_scfit)*(1-test_size))
    #scale all data to that of the projected training set datapoints
    X_fit = X_scfit[:fit_cutter]

    scaler1 = StandardScaler()
    scaler2 = RobustScaler()
    scaler3 = MinMaxScaler(feature_range=(-1,1))
    scaler1.fit(X_fit)
    return scaler1.transform(X_sctran)

#this function will need to be re-evaluated if 
#various test_sizes are used around different models
def reform_with_PCA_isolated(X_pcatran, X_pcafit, test_size, num_isol_feats, comps_PCA):
    fit_cutter_fit = int(len(X_pcafit)*(1-test_size))
    X_main_features_fit = X_pcafit[:, :-num_isol_feats]
    X_fit = X_main_features_fit[:fit_cutter_fit]#update fit values for PCA

    X_main_features = X_pcatran[:, :-num_isol_feats]
    X_time_features = X_pcatran[:, -num_isol_feats:]
    

    pca = PCA()

    pca = PCA(n_components = comps_PCA, random_state=0)
    pca.fit(X_fit)
    X_main_features = pca.transform(X_main_features)

    return np.hstack((X_main_features, X_time_features))





'''
def show_all_results(used_model, history, X_test, y_test):
    # LOSS
    epochs = range(1, len(history.history['loss']) + 1)
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, history.history['loss'], 'y', label='Training Loss')
    plt.plot(epochs, history.history['val_loss'], 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()
    # ACCURACY

    plt.plot(epochs, history.history['R2Score'], 'y', label='Training R2')
    plt.plot(epochs, history.history['val_R2Score'], 'r', label='Validation R2')
    plt.title('Training and Validation R2Score')
    plt.xlabel('Epoch')
    plt.ylabel('R2Score')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()


    #predicting the test set results
    y_pred = used_model.predict(X_test) 


    plt.scatter(y_pred, y_test, s=1)
    plt.axis('tight')
    plt.title('Testing Outputs')
    plt.xlabel('y_pred')
    plt.xlim(-.25,.25)
    plt.ylim(-.25,.25)
    plt.ylabel('y_test')
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = x_vals  # Since y = x
    plt.plot(x_vals, y_vals, '-', color='black', label='y = x', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0,color='black',linewidth=0.5)
    plt.show()

    #SCATTERPLOT #SCATTERPLOT  #SCATTERPLOT  #SCATTERPLOT  #SCATTERPLOT  #SCATTERPLOT  #SCATTERPLOT  #SCATTERPLOT  
    plt.scatter(y_pred, y_test, s=1)
    plt.grid()
    plt.axis('tight')
    plt.title('Testing Outputs')
    plt.xlabel('y_pred')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.ylabel('y_test')
    ax = plt.gca()
    x_vals = np.array(ax.get_xlim())
    y_vals = x_vals  # Since y = x
    plt.plot(x_vals, y_vals, '-', color='black', label='y = x', linewidth=0.5)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0,color='black',linewidth=0.5)
    plt.show()
    #DIRECTIONAL ACCURACY #DIRECTIONAL ACCURACY  #DIRECTIONAL ACCURACY  #DIRECTIONAL ACCURACY  #DIRECTIONAL ACCURACY  
    tp, fp, tn, fn = 0, 0, 0, 0
    tp5, fp5, tn5, fn5 = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if(y_pred[i]>0):
            if(y_test[i]>0):
                tp+=1
            if(y_test[i]<0):
                fp+=1
            if(y_pred[i]>=5):
                if(y_test[i]>0):
                    tp5+=1
                if(y_test[i]<0):
                    fp5+=1
        if(y_pred[i]<0):
            if(y_test[i]<0):
                tn+=1
            if(y_test[i]>0):
                fn+=1
            if(y_pred[i]<=-5):
                if(y_test[i]<0):
                    tn5+=1
                if(y_test[i]>0):
                    fn5+=1
    directionalAccuracy = ((tp+tn)/(tp+fp+tn+fn))*10000//1/100
    print('Directional Accuracy:\t\t',directionalAccuracy)
    directionalAccuracy5guess = ((tp5+tn5)/(tp5+fp5+tn5+fn5))*10000//1/100
    print('Directional Accuracy >(+/-)5:\t',directionalAccuracy5guess)
    '''