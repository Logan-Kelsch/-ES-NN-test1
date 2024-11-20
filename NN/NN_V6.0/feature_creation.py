'''
    Logan Kelsch
    This file is going to be used for the new data that will be collected.

    In V6.0, we will be SCRAPING only 6 features, instead of many.
    These 6 features will be:
    -   High, Low, Open, Close
    -   ToD (minute)
    -   Volume (of minute)

    With these features we will use this python file to create hopefully a few hundred features.
    These features will consist of a large sets of slightly altered dimensions of the provided data.
    
    ENGINEERED FEATURE CATEGORIES:
                        note-- price movement is naturally over time, consider terminology
                               with this in mind. (ex: velocity is displacement over time)
                        
                        notation: 
                                (description or term hints)
                                [value ranges low to high] %favorable_increments
                                {estimated total features}

    -   Price difference (velocity)             [1 - 60]                {60}
    -   Rate of price difference (acceleration) [1 - 60]                {60}
    -   Stochastic K, D, K-D (K is fast)        k=[5-60]%5   d=[k-60]%5 {78}
    -   RSI                                     [1 - 60]                {60}
    -   Bar height

    work in progress from here

    remaining old data labels and features:
    HL2,H2L,HLdiff12,HLdiff21,vol,vol10,vol15,FT,vol30,vol60,volD10,volD15,volD30,volD60,
    vpm5,vpm10,vpm15,vpm30,vpm60,ToD,DoW,mo,r1,r2,r3,r5,r10,r15,r30,r60

    ENGINEERED TARGET CATEGORIES:

    -   Price Difference
    -   Direction Classifications
    -   Volume Classifications
'''