'''
This file will be for the dataset creation of VHS variables after recent collection of understanding and utility.

'''

import pandas as pd
import numpy as np
from typing import Literal

def augmod_dataset(
	data	:	np.ndarray,
	mode	:	Literal['beta_testing_60_08'] = 'beta_testing_60_08'
):
	'''
	### info: ###
	This function will take a standard TOS cleaned csv file and transform it into a dataset of VHS stategy features.
	
	### params: ###
	- data:
	-	-	the dataset being brought in for modular augmentation
	- mode:
	-	-	what form of generation will be make of all featuresets
	
	### returns: ###
	An expanded dataset in pandas format
	'''
	
	#collect all desired feature sets

	v = fe_attention_hawkes_stoch(data, mode)
	h = fe_volatility_hawkes_stoch(data, mode)
	s = fe_direction_hawkes_stoch(data, mode)

	tod = fe_ToD(data)
	dow = fe_DoW(data)

	vel = fe_vel(data, mode)

	#concat the dataframes together, suppoosedly most efficient method for this with C backend
	vhs_df = pd.concat([data,tod,dow,v,h,s,vel], axis=1)
	
	#trim the lookback data that can loop, as well as look-forward data assuming target columns will be added
	trimmed_vhs_df = vhs_df.iloc[250:-60].reset_index(drop=True)

	#delete unused variables and return the dataframe
	del vhs_df, v, h, s
	return trimmed_vhs_df

#__________________ NOTE FEATURE ENGINEERING SECTION NOTE __________________#

def fe_attention_hawkes_stoch(data, mode):
	'''
	This function takes the stochastic range process of a hawkes process of some given attention function.<br>
	This attention function will be represented with market participation, as volume.
	'''

	match(mode):
		case 'beta_testing_60_08':
			lmbda = 60
			kappa = 0.08
		case 'feats_only':
			pass
		case 'feats_targets':
			pass
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")

	#the index of TOS cleaned data for volume is 3 (feats: h,l,c,v,t)
	vol = data.iloc[:,3].values

	kap = np.exp(-kappa)

	#creating empty arrays for futures values needed
	#actual hawkes values, not captured for algorithm return
	hwk_v = np.zeros(len(data), dtype=np.float32)
	#hawkes stochastic range process values, captured for algorithm return
	hsp_v = np.zeros(len(data), dtype=np.float32)

	#average and norm range collecting loop for attention
	for sample in range(240,len(data)-60):
		
		#localize the value to a variable to minimize referencing
		lcl_v = np.mean(vol[sample-lmbda:sample])
		
		#localize range to minimize referencing
		#ensure we dont have any hiccups with zero in denom
		if(lcl_v!=0):
			lcl_r = vol[sample]/lcl_v
		else:
			lcl_r = 0

		#localize the hawkes process value to minimize referencing
		#calculate hawkes process value for this sample
		lcl_h = hwk_v[sample-1] * kap + lcl_r

		#assign value to get it out of the way, and hawkes array needs to be referenced now
		hwk_v[sample] = lcl_h

		#collect highest and lowest from time range of hawkes process for stochastic range processing
		lcl_hsh = np.max(hwk_v[sample-lmbda:sample])
		lcl_hsl = np.min(hwk_v[sample-lmbda:sample])

		#ensuring that the range is not zero to avoid zero denom
		if(lcl_hsh-lcl_hsl != 0):
			#takes the location in 0-100 stochastic range of hawkes process values over lambda minutes
			hsp_v[sample] = (lcl_h-lcl_hsl) / (lcl_hsh-lcl_hsl) * 100
		else:
			hsp_v[sample] = 0

	#save memory??? no idea if python garbage collector gets this for me regardless
	del hwk_v

	#combine column names and feature set into a pandas dataframe and return to augmod or other function
	return pd.DataFrame(hsp_v, columns=fn_attention_hawkes_stoch(mode)).clip(lower=0,upper=100).round(2)

def fe_volatility_hawkes_stoch(data, mode):
	'''
	This function takes the stochastic range process of a hawkes process of some given volatility function.<br>
	This volatility function will be represented with market absolute velocity over time, True Range.
	'''

	match(mode):
		case 'beta_testing_60_08':
			lmbda = 60
			kappa = 0.08
		case 'feats_only':
			pass
		case 'feats_targets':
			pass
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")

	#the index of TOS cleaned data for high low and close are 0,1,2 (feats: h,l,c,v,t)
	h = data.iloc[:,0].values
	l = data.iloc[:,1].values
	c = data.iloc[:,2].values

	kap = np.exp(-kappa)

	#creating empty arrays for future values needed
	#actual true range values, not captured for algorithm return
	tru_rng = np.zeros(len(data), dtype=np.float32)
	#actual hawkes values, not captured for algorithm return
	hwk_v = np.zeros(len(data), dtype=np.float32)
	#hawkes stochastic range process values, captured for algorithm return
	hsp_v = np.zeros(len(data), dtype=np.float32)

	#average and norm range collecting loop for attention
	for sample in range(240,len(data)-60):
		
		#localize local candle height 
		lcl_ch = h[sample]-l[sample]

		#collect local true range value and assign immediately
		tru_rng[sample] = max(
			abs(lcl_ch),
			abs(h[sample]-c[(sample-1)]),
			abs(l[sample]-c[(sample-1)])
		)

		#localize the value to a variable to minimize referencing
		lcl_avg = np.mean(tru_rng[sample-lmbda:sample])
		
		#localize range to minimize referencing
		#ensure we dont have any hiccups with zero in denom
		if(lcl_avg!=0):
			lcl_rng = lcl_ch/lcl_avg
		else:
			lcl_rng = 0

		#localize the hawkes process value to minimize referencing
		#calculate hawkes process value for this sample
		lcl_h = hwk_v[sample-1] * kap + lcl_rng

		#assign value to get it out of the way, and hawkes array needs to be referenced now
		hwk_v[sample] = lcl_h

		#collect highest and lowest from time range of hawkes process for stochastic range processing
		lcl_hsh = np.max(hwk_v[sample-lmbda:sample])
		lcl_hsl = np.min(hwk_v[sample-lmbda:sample])

		#ensuring that the range is not zero to avoid zero denom
		if(lcl_hsh-lcl_hsl != 0):
			#takes the location in 0-100 stochastic range of hawkes process values over lambda minutes
			hsp_v[sample] = (lcl_h-lcl_hsl) / (lcl_hsh-lcl_hsl) * 100
		else:
			hsp_v[sample] = 0

	#save memory??? no idea if python garbage collector gets this for me regardless
	del hwk_v

	#combine column names and feature set into a pandas dataframe and return to augmod or other function
	return pd.DataFrame(hsp_v, columns=fn_volatility_hawkes_stoch(mode)).clip(lower=0,upper=100).round(2)

def fe_direction_hawkes_stoch(data, mode):
	'''
	This function takes the stochastic range process of a hawkes process of some given direction function.<br>
	This direction function will be represented with market stochastic range of price action.
	'''

	match(mode):
		case 'beta_testing_60_08':
			lmbda = 60
			kappa = 0.08
		case 'feats_only':
			pass
		case 'feats_targets':
			pass
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")

	#the index of TOS cleaned data for close is 2 (feats: h,l,c,v,t)
	c = data.iloc[:,2].values

	kap = np.exp(-kappa)

	#creating empty arrays for future values needed
	stoch = np.zeros(len(data), dtype=np.float32)
	#actual hawkes values, not captured for algorithm return
	hwk_v = np.zeros(len(data), dtype=np.float32)
	#hawkes stochastic range process values, captured for algorithm return
	hsp_v = np.zeros(len(data), dtype=np.float32)

	#average and norm range collecting loop for attention
	for sample in range(240,len(data)-60):
		
		#collect the highest and lowest values of the close in the last lmbda minutes
		s_hi = np.max(c[sample-lmbda:sample])
		s_lo = np.min(c[sample-lmbda:sample])

		#collect the actual stochastic values, can neglect *100 here
		if(s_hi-s_lo!=0):
			lcl_s = (c[sample] - s_lo) / (s_hi - s_lo)
		else:
			lcl_s = 0

		#localize the value to a variable to minimize referencing
		#lcl_avg = np.mean(stoch[sample-lmbda:sample])
		
		#localize range to minimize referencing
		#ensure we dont have any hiccups with zero in denom
		#if(lcl_avg!=0):
		#	lcl_rng = lcl_ch/lcl_avg

		#localize the hawkes process value to minimize referencing
		#calculate hawkes process value for this sample
		lcl_h = hwk_v[sample-1] * kap + lcl_s

		#assign value to get it out of the way, and hawkes array needs to be referenced now
		hwk_v[sample] = lcl_h

		#collect highest and lowest from time range of hawkes process for stochastic range processing
		lcl_hsh = np.max(hwk_v[sample-lmbda:sample])
		lcl_hsl = np.min(hwk_v[sample-lmbda:sample])

		#ensuring that the range is not zero to avoid zero denom
		if(lcl_hsh-lcl_hsl != 0):
			#takes the location in 0-100 stochastic range of hawkes process values over lambda minutes
			hsp_v[sample] = (lcl_h-lcl_hsl) / (lcl_hsh-lcl_hsl) * 100
		else:
			hsp_v[sample] = 0

	#save memory??? no idea if python garbage collector gets this for me regardless
	del hwk_v

	#combine column names and feature set into a pandas dataframe and return to augmod or other function
	return pd.DataFrame(hsp_v, columns=fn_direction_hawkes_stoch(mode)).clip(lower=0,upper=100).round(2)


def fe_vel(data, mode):

	match(mode):
		case 'beta_testing_60_08':
			lengths = [15,60]
		case 'feats_only':
			pass
		case 'feats_targets':
			pass
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")
		
	c = data.iloc[:,2].values

	vels = np.zeros((len(data),2), dtype=np.float32)

	for sample in range(240, len(data)-60):

		for l, length in enumerate(lengths):

			vels[sample,l] = c[sample] - c[sample-length]

	return pd.DataFrame(vels, columns=fn_vels(lengths)).round(2)


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


#__________________ NOTE feature name functions section NOTE _________________________#

def fn_vels(lengths):
	return [f'vel_{l}' for l in lengths]

def fn_attention_hawkes_stoch(mode):
	match(mode):
		case 'beta_testing_60_08':
			return ['v_60_.08']
		case 'feats_only':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case 'feats_targets':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")
	return

def fn_volatility_hawkes_stoch(mode):
	match(mode):
		case 'beta_testing_60_08':
			return ['h_60_.08']
		case 'feats_only':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case 'feats_targets':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")
	return

def fn_direction_hawkes_stoch(mode):
	match(mode):
		case 'beta_testing_60_08':
			return ['s_60_.08']
		case 'feats_only':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case 'feats_targets':
			raise NotImplementedError(f"Have not implemented mode {mode}.")
		case _:
			raise ValueError(f"Mode for augmod dataset was passed to a function with an illegal value of '{mode}'. Please check acceptable values and try again.")
	return

