'''
    Logan Kelsch
    This file is going to be used for the new data that will be collected.

    In V6.0, we will be SCRAPING only 6 features, instead of many.
    These 6 features will be:
    -   High, Low, Open, Close (model training data pre-processing removable)
    -   ToD (minute)
    -   Volume (of minute)
    - JUST ADDED 
    -   DoW

    With these features we will use this python file to create hopefully a few hundred features.
    These features will consist of a large sets of slightly altered dimensions of the provided data.
    
    ENGINEERED FEATURE CATEGORIES:
                        note-- price movement is naturally over time, consider terminology
                               with this in mind. (ex: velocity is displacement over time)
                        
                        notation: 
                                (description or term hints)
                                [value ranges low to high] %favorable_increments
                                {estimated total features}

        !! NOTE NOTE HUGE FINALIZATION NOTE: THERE SHOULD BE NO SUBTRACTION, ONLY DIVISION
        TO AVOID ACTUAL VALUE USAGE, IT SHOULD BE ACTUAL PERCENT USAGE SO ALL FUNCTIONS
        SHOULD LOOK SOMETHING LIKE feature = 100*(val1/val2-1) [this is representation of percent difference] 

    -   Price difference (velocity)             [1 - 60]                {60} 60
    -   Rate of price difference (acceleration) [1 - 60]                {60} 120
    -   Stochastic K, D, K-D (K is fast)        k=[5-60]%5   d=[k-60]%5 {78}    468
    -   RSI                                     [1 - 60]                {60} 180
    -   close - Moving average                  [1-60]%5 , [60-200]%20  {68}    536
    -   Mov-avg-diff                    [5, 10, 15, 30, 60, 120]        {30} 210
    -   close - lowest low                [5, 15, 30, 60, 120]          {5}
    -   close - highest high              [5, 15, 30, 60, 120]          {5}  220
    -   hihi - lolo                      [custom 2 pairs above]         {20} 240 
    -   bar height                                                      {1}
    -   wick height                                                     {1}
    -   uwick - lwick                                                   {1}     539
    -   high - (close, open, low)(, high holds) [1-5]                   {15} 255
    -   (high, close, open) - low(, low holds)  [1-5]                   {15} 270
    -   total volume                            [1-60]                  {60} 330
    -   total vol chunk difference              [1-60]%5                {11}    550
    -   volume - average volume                 [1-60]                  {60} 390

                                                550 features predicted

    remaining old data labels and features:
    HL2,H2L,HLdiff12,HLdiff21,vol,vol10,vol15,FT,vol30,vol60,volD10,volD15,volD30,volD60,
    vpm5,vpm10,vpm15,vpm30,vpm60,ToD,DoW,mo,r1,r2,r3,r5,r10,r15,r30,r60

    ENGINEERED TARGET CATEGORIES:

    -   Price Difference (will start here)
        -   price difference                    [1 - 60]                {60}

    NOTE can create these later
    -   Direction Classifications
    -   Volume Classifications
'''

'''
    NOTE ALL DATASETS HANDLED IN THESE FUNCTIONS WILL HANDLED AS PANDAS DATAFRAMES
         TO ALLOW FOR USAGE OF NAMES OF FEATURES/TARGET COLUMNS
'''

import pandas as pd
import numpy as np

# main function to create all features, takes in dataset
def generate_augmented_features(X):
    #X is assumed to come in as H, L, O, C, vol, ToD, DoW
    #This function will contain all functions to take or generate all sets of features




    return X


def generate_targets(X):
    #X is assumed to come in as H, L, O, C, vol, ToD, DoW
    y = None
    return y