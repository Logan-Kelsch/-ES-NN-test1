'''
-   ModelSet Training  - - Training all models on their set split of data
-   -   -   the double rotation model will be built first, an optimal-hyperparameter-set seeking algorithm should be built for this
-   -   -   LSTM NN will have to come later
'''
from _Hyperparam_Optimizer import *
from typing import Union, Literal
from sklearn.tree import DecisionTreeClassifier
from aeon.classification.sklearn import RotationForestClassifier


def show_available_model_types():
	'''printout for user to see all available model types and the strings to insert for them.'''
	print(f"\
	   'decision_tree'\n\
	   'dt'\n\
		-	-	Sci-Kit Learn DecisionTreeClassifier\n\
		-	-	{default_parameters('dt')}\n\
	   \n\
	   'rotation_forest'\n\
	   'aeon_rf'\n\
		   -	-	AEON Rotation Forest\n\
		-	-	{default_parameters('aeon_rf')}\n\
	   \n\
	   ")
	return

def default_parameters(model_type:str='')->dict:
	'''Return the dictionary of all keyword arguments for the model selected.
	Params:
	- model-type:
	-	_String indicating what type of model is used.
	'''
	if(model_type in ('dt','decision_tree')):
		return {'criterion':'gini','max_depth':8,'min_samples_split':2,'min_samples_leaf':1}
	elif(model_type in ('aeon_rf','rotation_forest')):
		return {'base_estimator':DecisionTreeClassifier(max_depth=4),'n_estimators':4,\
				  'min_group':1,'max_group':20,'remove_proportion':0.3,'n_jobs':-1}
	else:
		return {None:None}

def train_models(
	model_types	:	list	=	[]
	,data_parts :   list    =   []
	,trgt_parts	:	list	=	[]
	,param_mode	:	Literal['default','tuner','custom']	=	'default'
	,cst_mod_prm:	list	=	[]
	,tnr_verbose:	bool	=	True
):
	'''
		This function trains a set of models (identical type?) on a list of dataset partitions.
		Returns a 3D array of models NOTE CORRECT THIS reference ln275 NOTE for the 2d array of dataparts (is 2d no matter what)
		Params:
	- model_types:
	-	_A list of strings representing the types of models that will train
	- data-parts:
	-	_2D list of data training partitions, ALWAYS 2D!(D1:featurespaces, D2:samplespaces)
	- trgt-parts:
	-	_1D list of target training partitions,ALWAYS 1D!, only needs to follow samplespace partitions
	- param-mode:
	-	_Toggle for how parameters should be selected. default- 
	-	_-	default parameters (showable function call)
	-	_-	tuner parameters, uses hyper-parameter tuner to select params (high comp-cost)
	-	_-	custom parameters, use parameters entered from a list (for each model) of kwarg dicts
	- cst-mod-prm:
	-	_Custom Model Params, will act as alternative to hyperparameter_tuner. 
	-	_list of dicts for 1 or more than one model type
	- tnr-verbose:
	-	_Verbose option for the hyperparameter-tuner.
	'''

	#NOTE input validation  NOTE
	#NOTE  & corrections    NOTE ___________________________________________________________

	#Checking for empty list error
	if(len(model_types)<1):
		raise ValueError("FATAL: A list was provided for 'model_types', but nothing was in the list.")
	#ensuring a verbose printout for all other cases
	else:
		#if default parameters are to be used, say so and print out all parameters used.
		if(param_mode=='default'):
			print('Default parameters selected.')
			for mtype in model_types:#for each type, show type and parameters
				print(f'{mtype}	-	{default_parameters(mtype)}')
		#if custom parameters are to be used, say so and print out all parameters used
		elif(param_mode=='custom'):
			print('Custom parameters selected.')
			#ensure dimensions of parameters match dimensions of models, if not switch to default
			if(len(cst_mod_prm) != len(model_types)):
				print('NON-FATAL: Dimensions of Custom Parameters (cst_mod_prm) does not match \
					Dimensions of "Model_Types".\nALL MODEL PARAMETERS ARE FORCED TO DEFAULT.')
				#switch to default here and output all default parameters
				param_mode = 'default'
				for mtype in model_types:#for each type, show type and parameters
					print(f'{mtype}	-	{default_parameters(mtype)}')
			#if dimensions are correct output model types and all custom parameters
			else:
				for mt in range(len(model_types)):
					print(f'{model_types[mt]}	-	{cst_mod_prm[mt]}')
		#if the hyper-parameter tuner is selected, say so
		else:
			print('Hyper-parameter Tuner selected.')

	#NOTE end input val & cor NOTE __________________________________________________________

	#create 3D list of models to follow 2D array of training sets
	#the third dimension is the list of different models used.
	models = []

	#for each feature space (set of models and data)
	for feature_space in range(len(data_parts)):

		#the array of models built from this given featurespace
		feat_local_models = []

		#for each sample space (of this featurespace)
		for sample_space in range(len(data_parts[feature_space])):

			#this list should be of length model_types used!
			#all models here are trained on individual 
			smpl_local_models = []

			#current selected training set within the 2D array of training sets
			X_current_partition = data_parts[feature_space][sample_space]
			y_current_partition = trgt_parts[sample_space]

			#for each model within the list of model types, iterates next to cst_mod_prm
			for m in range(len(model_types)):
				
				'''NOTE-	-	-	-	-NOTE-	-	-	-NOTE-	-	-	-	-NOTE'''
				#NOTE Begin individual model creation and parameter setting END#NOTE#############################
				'''NOTE-	-	-	-	-NOTE-	-	-	-NOTE-	-	-	-	-NOTE'''

				#do match case for each type of model
				match(model_types[m]):
					
					#if the model is a decision tree
					case 'dt'|'decision_tree':
						
						#do match case for each parameter mode, default, tuner, custom
						match(param_mode):
		
							#if the parameter mode is set to default for each model
							case 'default':

								#see if keyword parameters will fit to model type
								try:
									model = DecisionTreeClassifier(**default_parameters(model_types[m]))
								#expecting typeErrors if any, printout and show here
								except TypeError as e:
									#expecting this specific error type, approach as follows
									if 'unexpected keyword argument' in str(e):
										raise TypeError(f'Default parameters: {default_parameters(model_types[m])}'
														f'Are not fitting to **kwargs for model of type {model_types[m]}')
									#all other typeErrors, still most likely case
									else:
										raise TypeError(f'FATAL: Unexpected TypeError:\n{e}')
								#any other unexpected error, unlikely but I would want to exit the program
								except Exception as e:
									print(f'FATAL: Unexpected error:\n{e}')
									raise
							
							#if the parameter mode is set to custom for each model
							case 'custom':

								#see if keyword parameters will fit to model type
								try:
									model = DecisionTreeClassifier(**cst_mod_prm[m])
								#expecting typeErrors if any, printout and show here
								except TypeError as e:
									#expecting this specific error type, approach as follows
									if 'unexpected keyword argument' in str(e):
										raise TypeError(f'Custom parameters: {cst_mod_prm[m]}'
														f'Are not fitting to **kwargs for model of type {model_types[m]}')
									#all other typeErrors, still most likely case
									else:
										raise TypeError(f'FATAL: Unexpected TypeError:\n{e}')
								#any other unexpected error, unlikely but I would want to exit the program
								except Exception as e:
									print(f'FATAL: Unexpected error:\n{e}')
									raise
							case 'tuner':
								#using try-except, (12/24/24) unsure of WHICH way this may fail
								try:
									model = hyperparameter_tuner(
										model_with_start_params=DecisionTreeClassifier(**default_parameters(model_types[m]))
										,tuner_verbose=tnr_verbose)
								#generalize exception-e statement as tuner error if it fails
								except Exception as e:
									raise print(f'FATAL: Tuner error:\n{e}')
	
					#if the model is aeon's rotation forest
					case 'aeon_rf'|'rotation_forest':
		 
						#do match case for each parameter mode, default, tuner, custom
						match(param_mode):
		
							#if the parameter mode is set to default for each model
							case 'default':

								#see if keyword parameters will fit to model type
								try:
									model = RotationForestClassifier(**default_parameters(model_types[m]))
								#expecting typeErrors if any, printout and show here
								except TypeError as e:
									#expecting this specific error type, approach as follows
									if 'unexpected keyword argument' in str(e):
										raise TypeError(f'Default parameters: {default_parameters(model_types[m])}'
														f'Are not fitting to **kwargs for model of type {model_types[m]}')
									#all other typeErrors, still most likely case
									else:
										raise TypeError(f'FATAL: Unexpected TypeError:\n{e}')
								#any other unexpected error, unlikely but I would want to exit the program
								except Exception as e:
									print(f'FATAL: Unexpected error:\n{e}')
									raise
							
							#if the parameter mode is set to custom for each model
							case 'custom':

								#see if keyword parameters will fit to model type
								try:
									model = RotationForestClassifier(**cst_mod_prm[m])
								#expecting typeErrors if any, printout and show here
								except TypeError as e:
									#expecting this specific error type, approach as follows
									if 'unexpected keyword argument' in str(e):
										raise TypeError(f'Custom parameters: {cst_mod_prm[m]}'
														f'Are not fitting to **kwargs for model of type {model_types[m]}')
									#all other typeErrors, still most likely case
									else:
										raise TypeError(f'FATAL: Unexpected TypeError:\n{e}')
								#any other unexpected error, unlikely but I would want to exit the program
								except Exception as e:
									print(f'FATAL: Unexpected error:\n{e}')
									raise
							
							#if the parameter mode is set to utilize the hyperparameter tuner
							case 'tuner':

								#using try-except, (12/24/24) unsure of WHICH way this may fail
								try:
									model = hyperparameter_tuner(
										model_with_start_params=RotationForestClassifier(**default_parameters(model_types[m]))
										,tuner_verbose=tnr_verbose)
								#generalize exception-e statement as tuner error if it fails
								except Exception as e:
									raise print(f'FATAL: Tuner error:\n{e}')
			
				'''NOTE-	-	-	-	-NOTE-	-	-	-NOTE-	-	-	-	-NOTE'''
				#NOTE Conclude individual model creation and parameter setting END#NOTE#############################
				'''NOTE-	-	-	-	-NOTE-	-	-	-NOTE-	-	-	-	-NOTE'''

				'''NOTE CONSIDER ADDING FITTING PARAMETERS FOR EACH MODEL TYPE, 
				   NOTE LIST-LIKE, ALIKE OTHERS PASSED IN. COULD BE USEFULE FOR
				   NOTE IF NEURAL NETWORK WILL BE USED.'''
				
				'''NOTE Consider adding functionality for sample weights and
				   NOTE class weights passed to this point? 
				   NOTE decision trees have sample weights &&
				   NOTE Neural Networds have class weights'''
				#fit each model to its correlating training partition set
				model.fit(X_current_partition, y_current_partition)

				#then append the model to the models local to this sample+feature space
				#these models are exact partition specific, each model here trained on same set
				smpl_local_models.append(model)

				#exiting modelspace loop
			#entering samplespace loop

			#collect and append all modelspaces of all samplespaces of THIS featurespace
			feat_local_models.append(smpl_local_models)

			#exiting samplespace loop
		#entering featurespace loop

		#collect and append each featurespace of all samplespaces of all modelspaces
		#into a final 3D (NO MATTA WHAT) list of models. dimensions:
		'''NOTE(D1:featurespace, D2:samplespace, D3:modelspace)NOTE'''
		models.append(feat_local_models)

		#exiting featurespace loop
	#entering end-of-function space

	#returns a 3D array of models, fit to their respective datasets!
	return