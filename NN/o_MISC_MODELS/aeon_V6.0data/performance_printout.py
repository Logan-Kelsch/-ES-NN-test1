'''
    This file is specifically for printout out the results of the model.
    This will be through the use of minimal functions that will allow
    full compatibility between model types, currently including:
            regression, binary and multiclass classification
'''

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import DecisionTreeClassifier
def new_dt(mt, max_depth):
    if(mt=='Regression'):
        return DecisionTreeRegressor(max_depth=max_depth)
    else:
        return DecisionTreeClassifier(max_depth=max_depth)

#function returns metric type as a string for printout as well as actual score
#metrics are accuracy are r2 for classifcation and regression respectively
def score_model(y_true, y_pred, threshold, params):
    if(params.model_type=='Classification'):
        y_pred = y_pred > threshold
        return 'ACCURACY', accuracy_score(y_true, y_pred)
    else:
        return 'R2-SCORE', r2_score(y_true, y_pred)

def param_score_search(model, params, X_train, y_train, X_test, y_test, dynamic_param_name, val_range):
    scores = []
    score_type = None
    print('value: ',sep='',end='')
    for i in val_range:
        print(i,end='..',sep='')
        if(dynamic_param_name=='max_depth'):
            setattr(model, 'base_estimator', new_dt(params.model_type, i))
        else:
            setattr(model, dynamic_param_name, i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score_type, score = score_model(y_test, y_pred, 0.5, params)
        scores.append(score)
    plt.plot(val_range, scores)
    plt.xlabel('Parameter Value')
    plt.ylabel(score_type)
    plt.title(f'{score_type} for values of {dynamic_param_name}')
    plt.show()

#function to print out loss function, should be universal
def graph_loss(epochs, history):
    plt.figure(figsize=(12, 6))
    plt.plot(epochs[1:], history.history['loss'][1:], 'black', label='Training Loss')
    plt.plot(epochs[1:], history.history['val_loss'][1:], 'red', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper right')
    plt.show()

#function that prints out all metrics requested
#params must come in as the class
def graph_metrics(params, epochs, history):
    if(params.model_type == 'Regression'):################
        # R2SCORE
        plt.plot(epochs[1:], history.history['R2Score'][1:], 'y', label='Training R2')
        plt.plot(epochs[1:], history.history['val_R2Score'][1:], 'r', label='Validation R2')
        plt.title('Training and Validation R2Score')
        plt.xlabel('Epoch')
        plt.ylabel('R2Score')
        plt.ylim(bottom=0)
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
    else: ### CLASSIFICATION ##############################
        # ACCURACY
        plt.plot(epochs[1:], history.history['accuracy'][1:], 'y', label='Training acc')
        plt.plot(epochs[1:], history.history['val_accuracy'][1:], 'r', label='Validation acc')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        # PRECISION
        plt.plot(epochs[1:], history.history['precision'][1:], 'y', label='Training Precision')
        plt.plot(epochs[1:], history.history['val_precision'][1:], 'r', label='Validation Precision')
        plt.title('Training and Validation Precision')
        plt.xlabel('Epoch')
        plt.ylabel('Precision')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()
        # RECALL
        plt.plot(epochs[1:], history.history['recall'][1:], 'y', label='Training Recall')
        plt.plot(epochs[1:], history.history['val_recall'][1:], 'r', label='Validation Recall')
        plt.title('Training and Validation Recall')
        plt.xlabel('Epoch')
        plt.ylabel('Recall')
        plt.legend(['Train', 'Validation'], loc='upper right')
        plt.show()

#function that graphs the predictions, this is either as 
#a scatter plot for regression outputs or
#a confusion matrix for classification outputs
def graph_predictions(y_pred, y_true, params, ds_name='NO_NAME'):
    if(params.model_type == 'Regression'):###########################
        plt.scatter(y_pred, y_true, s=1)
        plt.axis('tight')
        plt.title(f'{ds_name} Data Performance')
        plt.xlabel('y_pred')
        plt.xlim(-25,25)
        plt.ylim(-25,25)
        plt.ylabel('y_test')
        ax = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals  # Since y = x
        plt.plot(x_vals, y_vals, '-', color='black', label='y = x', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0,color='black',linewidth=0.5)
        plt.show()
        #SECOND GRAPH
        plt.scatter(y_pred, y_true, s=1)
        plt.grid()
        plt.axis('tight')
        plt.title(f'{ds_name} Data Performance')
        plt.xlabel('y_pred')
        plt.xlim(-10,10)
        plt.ylim(-10,10)
        plt.ylabel('y_test')
        ax = plt.gca()
        x_vals = np.array(ax.get_xlim())
        y_vals = x_vals  # Since y = x
        plt.plot(x_vals, y_vals, '-', color='black', label='y = x', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.axvline(0,color='black',linewidth=0.5)
        plt.show()
    else: ### CLASSIFICATION ######################################
        # Create the confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Plot the confusion matrix using seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', \
                    xticklabels=range(params.num_classes), yticklabels=range(params.num_classes))
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {params.num_classes}-Class Classification')
        plt.show()