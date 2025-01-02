'''
-   Model evaluation
-   -   -   Copy over the printout functions of previous versions
-   -   -   consider adding some sort of proba printouts?
'''

#NOTE WHEN REACHING INTO FEATURE INDICES USED AND ROTATION DECOMPOSER USED, THAT ARE SAVED IN
#NOTE THE PARALLEL ARRAY OF X_FEAT_FIND REMEMBER THAT IT IS A 1D ARRAY, AND IT ONLY RUNS PARALLEL 
#NOTE TO THE FEATURESPACE DIMENSION AND NOT THE SAMPLESPACE DIMENSION OR MODELSPACE DIMENSION, 
#NOTE SO REFER ONLY TO THE INDEX OF THE FEATURE DIMENSION WHEN REFERING TO IT.

'''
	watching a podcast and coding so just going to write notes.

	NOTE create a self test and independent test 3d array of performances for each model type
    NOTE predicament: how to show the overall performance of models?
    NOTE	-	options to show all performances
    NOTE	-	show full averages, std deviations, high and low
    NOTE	-	show type averages, std deviations, high and low (types: featurespace, samplespace, modelspace based)
    NOTE	-	show feat-samp pair averages, std devs, hi n' lo (defin: for each training set)

'''