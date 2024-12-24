'''
    In this function I want to be able to take in a base estimator.
    with this base estimator, use inpect to collect adjustable args.
    with args, I want to test +/- a passed threshold value (%?).
    I want to start with underfitting values, and search each adjustable param,
    using multithreading through pool for individual test ranges
    for the best score, adjust that parameter, and go again and again.
    NOTE this will theoretically stop when no single param adjustment alone
         can improve the model. This will be broad enough that I believe it
         will be strong enough. This is a greedy algorithm.
'''

import inspect
from multiprocessing import Pool

'''
    NOTE POOL EXAMPLE END#NOTE
    
from multiprocessing import Pool

# Define your function
def my_function(arg1, arg2):
    # Simulate some work
    return f"Processed: {arg1} and {arg2}"

# Prepare the sets of arguments
n = 5  # Number of processes (or tasks)
arguments = [(i, i * 2) for i in range(n)]  # List of argument tuples

# Use multiprocessing.Pool
if __name__ == '__main__':
    with Pool(processes=n) as pool:
        results = pool.starmap(my_function, arguments)

    print("Results:", results)

'''

'''
	NOTE INSPECT EXAMPLE END#NOTE
 
# Get the argument names
def get_argument_names(func):
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values()]

# Get arguments annotated as int
def get_int_arguments(func):
    signature = inspect.signature(func)
    return [param.name for param in signature.parameters.values() 
            if param.annotation is int]
'''


def hyperparameter_tuner(
        model_with_start_params:any=None
        ,tuner_verbose:bool=True
        )   ->  any:
    '''Model parameter should probably honestly be turned into a string, 
        and then types should be caught here
        and then the default parameters could be grabbed from the function
            predefined in modelset_training!'''
    return None