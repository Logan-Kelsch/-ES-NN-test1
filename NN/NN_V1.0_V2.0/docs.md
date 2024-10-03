________________________
_____9/15/24 Update_____
________________________

# 3.0 FEATURES INFO:
---info---
# Symbol: /ES:XCME
# Work Time: 4/10/23, 3:05 PM - 9/13/24, 1:55 PM
---features---
# fulltime- seconds since 1970 (for appending data)
# FullK-    fullK plot for Stochastic of past hour
# diffKD-   value difference between fullK/fullD of past hour
# OB , OS-  boolean if Stoch plots are in top/btm 20% of range
# vol-      trade volume of each bar
# s15,30,60-Current price/time slope of past n minutes 
#ToD-       Time of Day in seconds
#perc30,60- percentile of low and high of past 30,60 mins and day
#RSI-       RSI output of past hour
#Wpercent-  William Percent output of past hour
#acc-       Rate of acceleration of past hour over half hours
################################
#DESIRED OUTPUTS
#Are formulated by making a bull and bear 0-1 oscillator
#standard deviations derived from upcoming 4 hours of movement
#Value description:
#If value==0
#  the conditions are not appealing for next n minutes
#If value==1
#  the price moves >=1 std. dev. in that direciton for next n mins
#else
#  price moves y std. dev. in that direction for next n minutes
#bull/bear15 - 15 minute individual direction lookahead
#bull/bear30 - 30 minute individual direction lookahead
#bull/bear60 - 60 minute individual direction lookahead

______________________
____ORIGINAL ENTRY____
______________________

# To build a successful NN,

1. Import the data
2. Clean the data
3. Split the data into training/test sets
4. create a model
5. train a model
6. make predictions
7. evaluate and improve


## best data points to modify for collection:

- RSI
- STOCH
- MACD
- LIN REG
- WILLIAM POINTS CD
- D/V/A

## methods of dictating swing entry/exit:

- STOCH break/hold
- Vel Equilibrium
- MACD Equilibrium
