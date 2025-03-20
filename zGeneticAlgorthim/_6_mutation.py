#elite samples are exempt from mutation


#replace entire pattern with new pattern

import random

def mutation_round(
    shuffled_population:list=None,
    partial_mutation_prob:float=0.025,
    pattern_mutation_prob:float=0.025
):

    

    return

#this function generates boolean, true percent% of the time
def chance(
    percent:float=0.5
):

    #if percent comes in as integer
    if(percent>1):
        percent/=100

    #if chance falls within percent window
    if(random.random()<percent):
        return True
        
    else:
        return False