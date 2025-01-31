
'''
This file will be used for loading in a few different master models
and then collecting the popular vote for given samples.
The main difference between the master models is that they should be predicting different times in the future
'''

def chronos_predict(
		master_names    :   list    =   []
		,loading_kwargs	:	dict	=	{}
):
    '''
    ### In this function Im going to load in the test set within the function.
    ### I am then going to load in a seperate set of just the features,
	-	This should be used by all models
    I am then going to load in a set of ALL requested targets,
    -	find seperable pattern in csv for their names so
    -	the master-names variable can rip up target-set.
    -	in preprocessdata, implement targetset and also index 0+comparative only
    ### This target set should be indices parallel to master-names and loaded models
    ### Once all models are loaded in, all will predict and a set identical to lvl0predset format will be returned
    ### Make all master models depth 3, try pop vote then decide if its worth making deeper model (+target)
    after this the last thing to implement is final function (raw thinkorswim export into prediction.)
    - looks like: load_raw_to_fullset, chronos_predict, prediction_visualization then risk management development.
    '''