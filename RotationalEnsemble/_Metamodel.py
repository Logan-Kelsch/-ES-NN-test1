'''
-   Model Stacking and Prediction fusion
-   -   -   This could be done possibly by bagging or stacking methods already created
-   -   -   could also train an LSTM NN or other model on stacked model predictions for one best prediction
-   -   -   'meta-model'? could use linear regression for this or logistic regression
-   -   -   this I guess is called level-0 models vs level-1 model
'''

			#NOTE NOTE NOTE META MODEL IDEAS TO CONSIDER
			#TRY ALL. bi-log-reg / log-reg, naive-bayes, lin-reg, pop-vote, average, NN, LSTM NN, DT, timeseries DT
			#can start with just one, actually yeah just start with one BUT the option for all with notimplemented error for
			#incompleted methods, just to have a working method first! we are very close.

			#here is where different meta models will be created, can consider saving models at a different time,
			#im sure errors will arrise by this point, so try first to get up to and through meta model creation and performance
			#output before considering further, in execution should not take that long

from importlib import reload
import _Utility
import _Neural_Net
from typing import Literal
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from aeon.classification.interval_based import TimeSeriesForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train_test_meta_model(
	models			:	list
	,X_findx		:	list
	,X_trans		:	list
	,X_test
	,y_test
	,val_size		:	float	=	0.2
	,shuffle		:	bool	=	False
	,metam_type		:	Literal['popular_vote','logistic_regression','NN','knn'
						'timeseries_forest','decision_tree','dummy'] = 'popular_vote'
	,use_cls_wt		:	bool	=	True
	,use_mm_params	:	bool	=	False
	,metam_params	:	dict	=	None
	,prediciton_type:	Literal['Classification','Regression']		=	'Classification'
)   ->    list:
	'''
		This function takes in a 3D list of trained models and uses their predictions as
		features to train a level-1 (meta-model) on, as a method of prediction fusion.
	'''

	#Notes for this function
	#contains ==None comparison, may cause error with __eq__ override
	
	#boolean value for differentiating whether .fit is used (whether data is seen before predicting)
	nolearn_model = True if metam_type in ('popular_vote','dummy') else False

	#this variable will be a flattened list of all binary prediction values of each model
	#This list will also be synonymous to a transpose of a featureset for training
	binary_predictions = []
	#proba_predictions = []
	for f_index, fspace in enumerate(models):
		for sspace in fspace:
			for model in sspace:
				#NOTE HERE we are looking at an individual model HERE END#NOTE

				#the local set here is the X_test set rotated relative to each model
				#Explanation:
				#Use rotation function[relevant to model].to transform(test-set using only[:, features[that are model specific]])
				local_set = X_trans[f_index].transform(X_test[:, X_findx[f_index]])
				local_pred= model.predict(local_set)
				#print(local_pred)
				binary_predictions.append(local_pred)
				#proba_predictions.append(model.proba_predict())

	#create a reforomatted 'featureset' of the predictions to be used in any .fit application
	if(len(binary_predictions) < len(binary_predictions[0])):
		X_test = np.array(binary_predictions).T
		X_test = np.squeeze(X_test)
	#of course reformat the targets just for harmony, not sure if this is actually needed
	if(type(y_test)==list):
		y_test = np.array(y_test)

	print(X_test.shape)

	#Splitting the training set into what the metamodel is trained on, and what it is validated on.
	X_metatrain, X_metatest, y_metatrain, y_metatest = train_test_split(X_test, y_test, test_size=val_size, shuffle=shuffle)

	metamodel = meta_train(metam_type, X_metatrain, y_metatrain, X_metatest, y_metatest,
						use_class_weight=use_cls_wt, 
						metam_params= metam_params if use_mm_params else None,
						prediction_type=prediciton_type)

	#collect predictions for self and for independent testing, based on whether model peeks at data
	match(nolearn_model):
		case True:#Sugguests no data was seen by meta-model (ex: pop-vote)
			indp_pred = meta_predict(metam_type, metamodel, X_test)
			self_pred = None

			indp_true = y_test
			self_true = None

		case False:#Suggests data was seen by meta-model (ex: NN)
			self_pred = meta_predict(metam_type, metamodel, X_metatrain)
			indp_pred = meta_predict(metam_type, metamodel, X_metatest)

			self_true = y_metatrain
			indp_true = y_metatest

	#NOTE HERE we have collected all predictions and truths, output performances here.
	
	'''NOTE NOTE SELFTESET OUTPUT SELFTEST OUPUT NOTE NOTE'''
	#allowing self test to be skipped if model does not see anything
	if(not nolearn_model):
		print(f'META-MODEL SELF TEST:'
			  f'\n\tAccuracy:\t{round(accuracy_score(self_true, self_pred), 2)}'
			  f'\n\tPrecision:\t{round(precision_score(self_true, self_pred), 2)}'
			  f'\n\tRecall:\t\t{round(recall_score(self_true, self_pred), 2)}')
		#Create the confusion matrix
		cm = confusion_matrix(self_true, self_pred)
		# Plot the confusion matrix using seaborn
		plt.figure(figsize=(8, 6))
		sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', \
					xticklabels=range(2), yticklabels=range(2))
		plt.xlabel('Predicted')
		plt.ylabel('True')
		plt.title(f'Confusion Matrix for Meta-Model Self Test')
		plt.show()

	'''NOTE NOTE INDEPENDENT TEST INDEPENDENT TEST NOTE NOTE'''
	print(f'META-MODEL INDEPENDENT TEST:'
		  f'\n\tAccuracy:\t{round(accuracy_score(indp_true, indp_pred), 2)}'
		  f'\n\tPrecision:\t{round(precision_score(indp_true, indp_pred), 2)}'
		  f'\n\tRecall:\t\t{round(recall_score(indp_true, indp_pred), 2)}')
	#Create the confusion matrix
	cm = confusion_matrix(indp_true, indp_pred)
	# Plot the confusion matrix using seaborn
	plt.figure(figsize=(8, 6))
	sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', \
				xticklabels=range(2), yticklabels=range(2))
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.title(f'Confusion Matrix for Meta-Model Independent Test')
	plt.show()

	return metamodel, X_test
		
def meta_train(metam_type, X_train, y_train, X_test, y_test, use_class_weight, metam_params = None, prediction_type = 'Classification'):
	'''
	This function will be used to train and return a model.
	This function returns 'None' for any nolearner models
	'''

	#training is split here based off of type of meta model requested
	match(metam_type):

		#If the user requests model is popular vote, return None, nolearner model
		case 'popular_vote'|'dummy':
			model = None
		
		case 'logistic_regression':
			if(metam_params == None):
				model = LogisticRegression().fit(X_train, y_train)
			else:
				model = LogisticRegression(**metam_params).fit(X_train, y_train)
		
		case 'knn':
			if(metam_params == None):
				model = KNeighborsClassifier(n_neighbors=20).fit(X_train, y_train)
			else:
				model = KNeighborsClassifier(**metam_params).fit(X_train, y_train)

		case 'decision_tree':
			if(metam_params == None):
				model = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
			else:
				model = DecisionTreeClassifier(**metam_params).fit(X_train, y_train)

		case 'NN':
			model = _Neural_Net.NN(prediction_type)
			if(metam_params):
				model.build(X_train, y_train, **metam_params)
			else:
				model.build()#??? should be fine
			history = model.fit(X_train, y_train, X_test, y_test)
			

	#Here I am going to have a few cases to add class weights if requested.
	if(use_class_weight):
		#If the model is a base classifier with the class weight parameter
		if hasattr(model,'class_weight'):
				#apply the class weight
				model.class_weight = _Utility.get_class_weights(y_train)
		elif hasattr(model, 'base_estimator'):
			#If it does, then check if that base classifier has class weights
			if hasattr(model.base_estimator, 'class_weight'):
				#apply the class weight to the models base classifier if so
				model.base_estimator.class_weight = _Utility.get_class_weights(y_train)

	return model

def meta_predict(metam_type, metamodel, X_test):
	#first ensure that X_test is iterable
	try:
		iter(X_test)
	except TypeError:
		raise TypeError(f"FATAL: In meta_predict() construction of type {metam_type}, the X_test data passed in was not iterable.")

		#the path here is split based off of what type of meta model will be used, for unique .fit usage
	match(metam_type):
		
		#If the user requests the use of a popular vote, no training required..
		case 'popular_vote':

			#create the prediction array for return
			y_pred = np.array([])

			#go through each sample of the X_test set
			for sample in X_test:
				#using bucket method, count the predictions for each classificaiton and append the most popular prediction to y_pred
				pred_counts = np.bincount(sample)
				y_pred = np.append(y_pred, np.argmax(pred_counts))

			return y_pred

		#if the user requests pointless predictions, of value 1
		case 'dummy':

			#My dummy predictor in utility, also has custom prediction value
			y_pred = _Utility.dummy_predict(X_test)
			return y_pred

		#Logistic Regression Model is of the sklearn variety
		case 'logistic_regression':
			
			#prediction function prewired in sklearn, ready to go
			y_pred = metamodel.predict(X_test)
			return y_pred
		
		#knn is of the sklearn variety - k nearest neighbor
		case 'knn':

			#prediction function prewired in sklearn, ready to go
			y_pred = metamodel.predict(X_test)
			return y_pred
		
		case 'decision_tree':

			#prediction function prewired in sklearn, ready to go
			y_pred = metamodel.predict(X_test)
			return y_pred

		case 'NN':
			y_pred = metamodel.predict(X_test)
			return y_pred








'''
NOTE NOTE NOTE NOTE
GPT example metamodel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np

# Example base model predictions (n_samples, n_models)
base_model_preds = np.array([[0.2, 0.6, 0.8],
							  [0.9, 0.7, 0.3],
							  [0.1, 0.2, 0.5]])

# Actual labels
y = np.array([1, 0, 0])

# Split meta-data into train/test for stacking
X_train, X_test, y_train, y_test = train_test_split(base_model_preds, y, test_size=0.2, random_state=42)

# Train meta-model
meta_model = LogisticRegression()
meta_model.fit(X_train, y_train)

# Predict using meta-model
final_predictions = meta_model.predict(X_test)
print(final_predictions)

...
# Predict probabilities
y_prob = log_reg.predict_proba(X_test)[:, 1]  # Probability of class 1

# Predict binary outcomes (default threshold = 0.5)
y_pred = log_reg.predict(X_test)
...

END#NOTE END#NOTE END#NOTE'''