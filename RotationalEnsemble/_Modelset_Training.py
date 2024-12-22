'''
-   ModelSet Training  - - Training all models on their set split of data
-   -   -   the double rotation model will be built first, an optimal-hyperparameter-set seeking algorithm should be built for this
-   -   -   LSTM NN will have to come later
'''
from typing import Union

def train_models(
	model_types	:	list	=	[]
	,data_parts :   list    =   []
	,cst_mod_prm:	list	=	[]
):
	'''
		This function trains a set of models (identical type?) on a list of dataset partitions.
		Returns a 2d array of models for the 2d array of dataparts (is 2d no matter what)
		Params:
	- model_types:
	-	_A list of strings representing the types of models that will train
	- data-parts:
	-	_2D array of data training segments, is 2D no matter what, (D1 is featurespaces, D2 is samplespaces)
	- cst_mod_prm:
	-	_Custom Model Params, will act as alternative to hyperparameter_tuner. 
	-	_list of dicts for 1 or more than one model type

	'''
	#Checking for empty list error
	if(len(model_types)<1):
		raise ValueError("FATAL: A list was provided for 'model_types', but nothing was in the list.")

	#create 2D list of models to follow 2D array of training sets
	models = []

	for feature_space in range(len(data_parts)):

		#
		feat_local_models = []

		for sample_space in range(len(data_parts[feature_space])):

			#this list should be of length model_types used!
			#all models here are trained on individual 
			smpl_local_models = []

			example_training_set = data_parts[feature_space][sample_space]

			for m in range(len(model_types)):

				if(model_types[m] == 'aeon_rotation_forest'):
					model = RotationForestClassifier()




					'''
					# List of tuples
# Dictionary of parameters
params = {
    "max_depth": 5,
    "random_state": 42
}

# Function that accepts parameters as keyword arguments
def decision_tree(max_depth=None, random_state=None):
    print(f"max_depth: {max_depth}, random_state: {random_state}")

# Call the function using the dictionary
decision_tree(**params)


					'''
					

	return