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

        ##NOTE TEMPORARY -- never mind
        
    # NOTE indicates complete

    -   Derivative of (looping index only) feature set A set
    -   #Price difference (velocity)             [1 - 60]                {60} 60
    -   #Rate of price difference (acceleration) [1 - 60]                {60} 120
    -   Stochastic K, D, K-D (K is fast)        k=[5-60]%5   d=[k-60]%5 {78}    468
    -   -   D is just moving average, come back to this, K-D can be same function
    -   #-   K
    -   RSI                                     [1 - 60]                {60} 180
    -   #close - Moving average                  [1-60]%5 , [60-240]%20  {68}    536
    -   #Mov-avg-diff                    [5, 10, 15, 30, 60, 120]        {30} 210
    -   #close - lowest low                [5, 15, 30, 60, 120]          {5}
    -   #close - highest high              [5, 15, 30, 60, 120]          {5}  220
    -   #hihi - lolo                      [custom 2 pairs above]         {20} 240 
    -   hilo stoch
    -   #bar height                                                      {1}
    -   #wick height                                                     {1}
    -   #uwick - lwick                                                   {1}     539
    -   high - (close, open, low)(, high holds) [1-5]                   {15} 255
    -   (high, close, open) - low(, low holds)  [1-5]                   {15} 270
    -   #total volume                            [1-60]                  {60} 330
    -   total vol chunk difference              [1-60]%5                {11}    550
    -   #volume - average volume                 [1-60]                  {60} 390
    -   consider h - l 1min vel with moving averages
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

times = [5,15,30,60,120,240]

######### '''NOTE NOTE''''''NOTE NOTE''' #########
###* * MOST IMPORTANT FUNCTION IN THIS FILE * *###
######______________________________________######

#this function will take in a dataset and generate 
#all requested features sets as well as target sets
#the output will be a pandas dataframe, fully concatenated
def augmod_dataset(data):
    
    #FEATURE ENGINEERING
    f_vel = fe_vel(data)#set
    f_acc = fe_acc(data)#set
    f_stchK = fe_stoch_k(data)#set
    f_ToD = fe_ToD(data)#single
    f_DoW = fe_DoW(data)#single
    f_barH = fe_height_bar(data)#single
    f_wickH = fe_height_wick(data)#single
    f_wickD = fe_diff_hl_wick(data)#single
    f_volData = fe_vol_sz_diff(data)#set
    f_maData = fe_ma_disp(data)#set
    f_maDiff = fe_ma_diff(f_maData)#set
    f_lolo = fe_lolo_diff(data)#set
    f_hihi = fe_hihi_diff(data)#set
    f_hilo = fe_hilo_diff(f_hihi,f_lolo)#set
    f_stochHiLo = fe_hilo_stoch(data, f_hihi, f_lolo)#set

    #TARGET ENGINEERING
    targets = te_vel(data)

    #list of dataframes
    df_list = [data, f_ToD, f_DoW, f_vel, f_acc, \
               f_stchK, f_barH, f_wickH, f_wickD,\
                f_volData, f_maData, f_maDiff, f_hihi, f_lolo,\
                    f_hilo, f_stochHiLo, targets]

    #cut off error head and error tail of dataframes
    df_trunk_1 = [df.iloc[:-60] for df in df_list]
    df_trunk_2 = [df.iloc[240:] for df in df_trunk_1]

    #concat all dataframes into one parallel set
    full_augmod = pd.concat(df_trunk_2, axis=1)

    return full_augmod

#this function returns the set of all names that are being requested.
#these are feature names that will likely be used to drop from the used dataset
def return_name_collection():

    set1 = fn_hilo_prices()
    set2 = fn_ma_prices()
    set3 = fn_orig_price()

    full_set = set1+set2+set3

    return full_set

'''-------------------------------------------------------------------------------
    NOTE FEATURE SPECIFIC FUNCTIONS
    NOTE fe_ denotes 'feature engineering'
'''#------------------------------------------------------------------------------

#returns lowest close of different ranges
#and the close difference to each
#this function requires cutting first 240 samples
def fe_lolo_diff(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for sample in range(l):
        row = []

        #getting lowest lows
        for dist in times:
            if(sample-dist < 0):
                row.append(0)
            else:
                lolo = np.min(close[sample-dist:sample])
                row.append(lolo)

        #getting lolo close displacements
        for i in range(len(times)):
            disp = close[sample] - row[i]
            row.append(disp)

        new_data.append(row)

    cols = [f'lolo{i}' for i in times]+[f'disp_lolo{i}' for i in times]

    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

#returns highest close of different ranges
#and the close difference to each
#this function requires cutting first 240 samples
def fe_hihi_diff(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for sample in range(l):
        row = []

        #getting lowest lows
        for dist in times:
            if(sample-dist < 0):
                row.append(0)
            else:
                hihi = np.max(close[sample-dist:sample])
                row.append(hihi)

        #getting lolo close displacements
        for i in range(len(times)):
            disp = close[sample] - row[i]
            row.append(disp)

        new_data.append(row)

    cols = [f'hihi{i}' for i in times]+\
        [f'disp_hihi{i}' for i in times]

    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

#returns vol*time area and avg vol difference
#thsi function requires cutting first 60 samples 
def fe_vol_sz_diff(X):
    #orig feature #4
    # # # deals with volume of each minute
    volume = X.iloc[:, 3].values
    new_data = []

    l = len(X)
    for sample in range(l):
        row = []
        #creating 59 areas of total volume from 2 -> 60 minutes
        for i in range(1,60):
            t_vol = volume[sample]
            for j in range(1,i+1):
                t_vol+=volume[(sample - j) %l]
            row.append(t_vol)
        
        #creating 59 diffs for vol - avgvol from 2 -> 60 minutes
        for i in range(1,60):
            avg_vol = row[i-1]/(i+1)
            row.append(round(volume[sample] - avg_vol, 2))

        #this is all data for each given sample
        new_data.append(row)

    #custom feature name
    cols = [f'vol_m{i}' for i in range(2,61)]+\
        [f'vol_avgDiff{i}' for i in range(2,61)]

    #CONTINUE HERE THERE ARE ONLY 59 FEATURES
    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

#returns moving averages and close-ma difference
#this function requires cutting of first 240 samples
def fe_ma_disp(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for sample in range(l):
        row = []
        '''first create price MAs'''
        # 2-59 total mins, 58 cases
        for i in range(1,59):
            avg_price = close[sample]
            for j in range(1,i+1):
                avg_price+= close[(sample - j) %l]
            #convert price*time to an average
            row.append(round(avg_price / (i+1),2))
        # 60-240 %20 total mins, 10 cases 
        for i in range(59,240,20):
            avg_price = close[sample]
            for j in range(1,i+1):
                avg_price+= close[(sample - j) %l]
            #convert price*time to an average
            row.append(round(avg_price / (i+1),2))
        '''second create MA-close diffs'''
        for ma in range(68):
            ma_disp = round(close[sample] - row[ma],2)
            row.append(ma_disp)
        
        new_data.append(row)

    cols = [f'ma{i}' for i in range(2,60)]+\
           [f'ma{i}' for i in range(60,241,20)]+\
           [f'disp_ma{i}' for i in range(2,60)]+\
           [f'disp_ma{i}' for i in range(60,241,20)]
    
    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set


#return Time of Day in minutes
#this function requires no cutting
def fe_ToD(X):
    #orig feature #5
    # # # deals with time since 1/1/1970@12:00am in seconds
    full_time = X.iloc[:, 4].values
    new_data = []

    l = len(X)
    for sample in range(l):
        '''
            take full time
            minus time zone adjustment
            modulate total seconds around days
            convert into minutes
        '''
        tod = (((full_time[sample] - 18000) % 86400) / 60)
        new_data.append(tod)
    
    feature = pd.DataFrame(new_data, columns=['ToD'])

    return feature

#returns Day of Week (0-7 Sun-Sat, 1-5 Mon-Fri)
#this function requires no cutting
def fe_DoW(X):
    #orig feature #5
    # # # deals with time since 1/1/1970@12:00am in seconds
    full_time = X.iloc[:, 4].values
    new_data = []

    l = len(X)
    for sample in range(l):
        '''
            take full time
            minus time zone and week adjustments
            convert into days
            modulate total days around weeks
            floor division for integer output
        '''
        dow = ((((full_time[sample] - 277200) / 86400) % 7) // 1)
        new_data.append(dow)

    feature = pd.DataFrame(new_data, columns=['DoW'])

    return feature

#difference of upper/lower wick size
#this function requires cutting first 1 sample
def fe_diff_hl_wick(X):
    new_data = []
    #get high, low, close values
    low = X.iloc[:, 1].values
    high = X.iloc[:, 0].values
    close = X.iloc[:, 2].values

    l = len(X)
    for sample in range(l):
        #height of upper wick
        u_wick = high[sample] - close[sample]
        #height of lower wick
        l_wick = close[(sample - 1) %l] - low[sample]

        new_data.append(u_wick - l_wick)

    feature = pd.DataFrame(new_data, columns=['diff_wick'])

    return feature


#high of candle (bar)
#this function requires cutting first 1 sample
def fe_height_bar(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for sample in range(l):
        #abs difference from close and open
        h = abs(close[sample] - close[(sample - 1) %l])
        new_data.append(h)

    feature = pd.DataFrame(new_data, columns=['barH'])

    return feature

#height of wicks and bar
#this function requires cutting first 1 sample
def fe_height_wick(X):
    #orig feature #1, #2
    # # # deals with open and close
    high = X.iloc[:, 0].values
    low = X.iloc[:, 0].values
    new_data = []

    l = len(X)
    for sample in range(l):
        #high minus low of each candle/wick
        h = high[sample] - low[sample]
        new_data.append(h)

    feature = pd.DataFrame(new_data, columns=['wickH'])

    return feature

#velocities
#this function requires cutting first 60 samples (df.iloc[60:])
def fe_vel(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for sample in range(l):
        row = []
        for displace in range(1,61):
            row.append(close[sample %l] - close[(sample-displace) %l])
        new_data.append(row)
    
    feature_set = pd.DataFrame(new_data, columns=[f'vel{i}' for i in range(1,61)])

    return feature_set


#accelerations
#this function requires cutting first 61 samples (df.iloc[61:])
def fe_acc(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for i in range(l):
        row = []
        for displace in range(1,61):
            # Calculate i + feature_num and handle out-of-bounds by wrapping around using modulo
            j = (i - displace)
            #actual value in csv is 100th of percent move
            vel1 = close[(i-1)  %l] - close[(j-1)   %l]
            vel2 = close[i      %l] - close[j       %l]
            row.append(vel2-vel1)
        new_data.append(row)
    # Convert to a new DataFrame
    feature_set = pd.DataFrame(new_data, columns=[f'acc{i+1}' for i in range(60)])

    #print(feature_set)
    return feature_set

#Stochastic K ONLY
#this function requires cutting first 120 samples (df.iloc[120:])
#used zero-out method for Null values instead of looping.
def fe_stoch_k(X):

    new_data = []
    #get high, low, close values
    low = X.iloc[:, 1].values
    high = X.iloc[:, 0].values
    close = X.iloc[:, 2].values
    i = 0

    l = len(X)
    for sample in range(l):
        row = []
        for i in range(5,125,5):
            #zero-out to avoid segmentation bound error
            if(sample-i<0):
                row.append(0)
            else:
                lowest_k = np.min(low[sample-i:sample])
                c1 = close[sample] - lowest_k
                c2 = np.max(high[sample-i:sample]) - lowest_k
                k = 0
                if(c2!=0):
                    k = c1/c2*100
                row.append(round(k,2))
        new_data.append(row)
    
    features_set = pd.DataFrame(new_data, columns=[f'stchK{i}' for i in range(5, 125, 5)])
    
    return features_set
            
#this function is directly interacting with collected stochK data
#this will be a lot of features if working with k values
#up to two hours old, 
#NOTE will come back to this later if wanted
def fe_stoch_d(f_stochK):

    new_data = []

    return new_data

#function return the set of ma differences from a set
#this function requires cutting first 120 samples
def fe_ma_diff(maData):

    #all used moving averages
    ma5 = maData.iloc[:, 3].values
    ma15 = maData.iloc[:, 13].values
    ma30 = maData.iloc[:, 28].values
    ma60 = maData.iloc[:, 58].values
    ma120 = maData.iloc[:, 61].values
    ma240 = maData.iloc[:, 67].values

    #array of arrays for ease of access
    mas = [ma5, ma15, ma30, ma60, ma120, ma240]

    new_data = []

    l = len(maData)
    for sample in range(l):
        row = []
        #nested to access two ma values at once for comparison
        for i in range(len(mas)):
            for j in range(i+1,len(mas)):
                ma_1 = mas[i][sample]
                ma_2 = mas[j][sample]
                row.append(round(ma_1 - ma_2, 2))
        new_data.append(row)
    
    cols = []
    lengths = [5,15,30,60,120,240]

    #prepping the feature names according to ma's used
    for i in range(len(lengths)):
        for j in range(i+1,len(lengths)):
            cols.append(f'diff_ma_{lengths[i]}_{lengths[j]}')

    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

#function return the differences between hihi and lolo of time sets
#this function requires cutting first 240 samples
def fe_hilo_diff(hihi_data, lolo_data):

    lengths = [5,15,30,60,120,240]
    
    hihi = hihi_data.values
    lolo = lolo_data.values

    new_data = []

    l = len(hihi)
    for sample in range(l):
        row = []
        #nested to access two ma values at once for comparison
        for i in range(len(lengths)):
            for j in range(len(lengths)):
                hi = hihi[sample, i]
                lo = lolo[sample, j]
                row.append(round(hi - lo, 2))
        new_data.append(row)
    
    cols = []

    #prepping the feature names according to ma's used
    for i in range(len(lengths)):
        for j in range(len(lengths)):
            cols.append(f'diff_hilo_{lengths[i]}_{lengths[j]}')

    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

#function returns location percent (like stochastic) between hihi lolo for each
#this function requires cutting first -- samples
def fe_hilo_stoch(X, hihi_data, lolo_data):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values

    #   j       -nested in-   i       
    #low ranges -nested in- high ranges
    hihi = hihi_data.values
    lolo = lolo_data.values

    new_data = []

    l = len(X)
    #for each sample
    for sample in range(l):
        row = []
        for i in range(len(times)):
            for j in range(len(times)):
                lowest_k = np.min(lolo[sample, j])
                c1 = close[sample] - lowest_k
                c2 = np.max(hihi[sample, i]) - lowest_k
                k = 0
                if(c2!=0):
                    k = c1/c2*100
                row.append(round(k,2))
        new_data.append(row)
    cols = []

    #prepping the feature names according to ma's used
    for i in range(len(times)):
        for j in range(len(times)):
            cols.append(f'hilo_stoch_{times[i]}_{times[j]}')

    feature_set = pd.DataFrame(new_data, columns=cols)

    return feature_set

'''-------------------------------------------------------------------------------
    NOTE TARGET SPECIFIC FUNCTIONS
    NOTE te_ denotes 'target engineering'
'''#------------------------------------------------------------------------------

#simple price difference for 1-60 minutes 
#this function requires cutting first 60 samples (df.iloc[60:])
def te_vel(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for i in range(l):
        row = []
        for displace in range(1,61):
            row.append(close[(i + displace) %l] - close[i %l])
        new_data.append(row)
    # Convert to a new DataFrame
    feature_set = pd.DataFrame(new_data, columns=[f't_{i+1}' for i in range(60)])

    #print(feature_set)
    return feature_set

#simple classification set for 1-60 minutes
#this function requires cutting first 60 samples
def te_vel_class_d5(X):
    #orig feature #3
    # # # deals with all close of minute values
    close = X.iloc[:, 2].values
    new_data = []

    l = len(X)
    for i in range(l):
        row = []
        for displace in range(1,61):
            movement = close[(i + displace) %l] - close[i %l]
            c = np.sign(movement) + 1
            mag = 0
            if(abs(movement) >= 5):
                mag=1
            c+=mag
            row.append(c)
        new_data.append(row)
    # Convert to a new DataFrame
    feature_set = pd.DataFrame(new_data, columns=[f'tc_5_{i+1}' for i in range(60)])

    #print(feature_set)
    return feature_set

'''-------------------------------------------------------------------------------
    NOTE FEATURE/TARGET NAME FUNCTIONS
    NOTE fn_ denotes 'feature(/set) names' for mass feature dropping
    NOTE tn_ denotes 'target (/set) names' for mass target  dropping
'''#------------------------------------------------------------------------------

#fe_vel
#fe_acc
#fe_stoch_k
#fe_ToD
#fe_DoW
#fe_height_bar
#fe_height_wick
#fe_diff_hl_wick
#fe_vol_sz_diff
#fe_ma_disp
#fe_ma_diff
#fe_lolo_diff
#fe_hihi_diff
#fe_hilo_diff
def fn_hilo_diff():
    lengths = [5,15,30,60,120,240]
    cols = []

    #prepping the feature names according to ma's used
    for i in range(len(lengths)):
        for j in range(len(lengths)):
            cols.append(f'diff_hilo_{lengths[i]}_{lengths[j]}')
    
    return cols

def fn_orig_price():
    return ['high','low','close']

def fn_orig_vol():
    return ['volume']

def fn_orig_time():
    return ['time']

def fn_hilo_prices():
    fn_hihi = [f'hihi{i}' for i in times]
    fn_lolo = [f'lolo{i}' for i in times]
    cols = fn_hihi+fn_lolo
    return cols

def fn_ma_prices():
    cols = [f'ma{i}' for i in range(2,60)]+\
           [f'ma{i}' for i in range(60,241,20)]
    return cols