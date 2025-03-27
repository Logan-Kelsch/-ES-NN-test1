from typing import Literal
import pandas as pd
import sys
import traceback

def get_fss_aslists(
    source	:	Literal['form_519']	=	'form_519'
):
    
	match(source):

		case 'form_519':

			return NotImplementedError(f"DANG I GOTTA FIGURE OUT HOW TO GET THE DANG INDICES")
		
		case _:

			raise NotImplementedError(f"DANG YOU USING A SOURCE FOR FEATURE SUBSET INDICES THAT ISNT IMPLEMENTED.")

def load_large_csv(
	file_name	:	str	=	''
):
	print("Trying to load CSV file into DataFrame...")
 	#Attempt to load in pandas file
	try:
		chunks = []
		iter = 1
		for chunk in pd.read_csv(file_name, chunksize=25000):
			isoc = sys.getsizeof(chunk) #initial size of chunk
			print(f'loaded chunk {iter} of size: {isoc}')
			chunks.append(chunk)
			iter+=1
		print('concat chunks')
		data = pd.concat(chunks)
		print('concatted chunks')
		#variable for later printout
		print(f"Success.\nSize of dataset:\t{sys.getsizeof(data)}")
		#trying to convert all of the float64 to float32 for memory
		#float_cols = data.select_dtypes(include=['float64']).columns
	except Exception as e:
		#error output and traceback
		print(f'\nCould not load file ({file_name}). Please check the file name.')
		traceback.print_exc()
		raise

	return data

def drop_all_targets(
	dataset	:	any	=	None,
	source	:	Literal['form_519'] = 'form_519'
):
	
	match(source):

		case 'form_519':

			pass
	return#do something

def get_fss_from_value(
	fss	:	list,
	index:	int
):
	'''
	This function takes a given index and checks a fss <br>
	index collection and returns the index collection it is from
	'''
	
	iswhere = []

	for fs in fss:
		if(index in fs):
			iswhere = fs
			break

	return iswhere




































def get_full_feature_dict(fssd):
	combined = {}
	for d in fssd:
		combined.update(d)
	return combined



def fn_all_subsets(real_prices: bool = False, indices:int=-1, keep_time:bool=True):
	'''
	This function creates a list of lists for each subsection of feature types.
	I'm typing this as I am implementing the second index data, this is getting pretty complex.
	Will probably have to look for a more organized method of sorting. Godspeed

	Params:
	- real-prices:
	-	boolean to allow for the original high,low,close,vol,time to be included as features
	- indices:
	-	option to pick how feature subsets are split based off of which index the data is coming from.
	-	 0) first index
	-	 1) second index
	-	-1) all indices as combined set 
	-	-2) all indices as seperate sets
	'''
	#feature name subsets
	fnsub = []
	# will append each individual feature/f_set here
	if(real_prices):
		if(indices == 0):
			fnsub.append(['high','low','close'])
			fnsub.append(['volume'])
			fnsub.append(['barH_spx','wickH_spx','diff_wick_spx'])
		'''elif(indices == 1):
			if(keep_time):
				fnsub.append(['high.1','low.1','close.1','time','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
		elif(indices == -1):
			if(keep_time):
				fnsub.append(['high','low','close','time','volume',\
						'high.1','low.1','close.1','volume.1',\
						'ToD','DoW','barH_spx','wickH_spx','diff_wick_spx',\
								'barH_ndx','wickH_ndx','diff_wick_ndx'])#removable real prices
			else:
				fnsub.append(['high','low','close','volume',\
						'high.1','low.1','close.1','volume.1',\
						'ToD','DoW','barH_spx','wickH_spx','diff_wick_spx',\
								'barH_ndx','wickH_ndx','diff_wick_ndx'])#removable real prices
		elif(indices == -2):
			if(keep_time):
				fnsub.append(['high','low','close','time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['high','low','close','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['high.1','low.1','close.1','volume.1','barH_ndx','wickH_ndx','diff_wick_ndx'])'''
	else:
		raise NotImplementedError(f"why are we removing real prices brah?")
		if(indices == 0):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
		'''if(indices == 1):
			if(keep_time):
				fnsub.append(['time','volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
		if(indices ==-1):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx','barH_ndx','wickH_ndx','diff_wick_ndx'])
		if(indices ==-2):
			if(keep_time):
				fnsub.append(['time','volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])
			else:
				fnsub.append(['volume','ToD','DoW','barH_spx','wickH_spx','diff_wick_spx'])
				fnsub.append(['volume.1','ToD','DoW','barH_ndx','wickH_ndx','diff_wick_ndx'])'''
		
	if(indices == 0 or indices ==-2):
		#		NOTE NOTE NOTE HERE IS THE IMPLEMENTATION OF ALL INDEX #1 (SPX) DATA. END NOTE END NOTE END NOTE		#
		fnsub.append(fn_vel(0))
		fnsub.append(fn_acc(0))
		fnsub.append(fn_stoch_k(0))
		fnsub.append(fn_vol_m(0))
		fnsub.append(fn_vol_avgDiff(0))
		if(real_prices):
			fnsub.append(fn_ma_s60(0))#removable
			fnsub.append(fn_ma_s240(0))#removable
		fnsub.append(fn_disp_ma60(0))
		fnsub.append(fn_disp_ma240(0))
		fnsub.append(fn_ma_diff(0))
		if(real_prices):	
			fnsub.append(fn_hihi(0))#removable
		fnsub.append(fn_disp_hihi(0))
		if(real_prices):	
			fnsub.append(fn_lolo(0))#removable
		fnsub.append(fn_disp_lolo(0))
		fnsub.append(fn_hilo_diff(0))
		fnsub.append(fn_hilo_stoch(0))

	if(indices == 1 or indices ==-2):
		#		NOTE NOTE NOTE HERE IS THE IMPLEMENTATION OF ALL INDEX #2 (NDX) DATA. END NOTE END NOTE END NOTE 		#
		fnsub.append(fn_vel(1))
		fnsub.append(fn_acc(1))
		fnsub.append(fn_stoch_k(1))
		fnsub.append(fn_vol_m(1))
		fnsub.append(fn_vol_avgDiff(1))
		if(real_prices):
			fnsub.append(fn_ma_s60(1))#removable
			fnsub.append(fn_ma_s240(1))#removable
		fnsub.append(fn_disp_ma60(1))
		fnsub.append(fn_disp_ma240(1))
		fnsub.append(fn_ma_diff(1))
		if(real_prices):	
			fnsub.append(fn_hihi(1))#removable
		fnsub.append(fn_disp_hihi(1))
		if(real_prices):	
			fnsub.append(fn_lolo(1))#removable
		fnsub.append(fn_disp_lolo(1))
		fnsub.append(fn_hilo_diff(1))
		fnsub.append(fn_hilo_stoch(1))

	if(indices == -1):
		raise NotImplementedError(f"FATAL: inclusive set building for featuresets has not been implemented for general feature-sets. 0,1,-2 (1/26/25) values are working.")
	
	return fnsub















times = [5,15,30,60,120,240]
idx	  = ['spx','ndx','NULL','NULL','NULL','ndx']




def fnsubset_to_indexdictlist(pddf_features, fnsub):
	'''This function takes: 
	-   a dataframe 
	-   list of lists of feature names
	  and turns it into 
	-   a list of dicts
	  The dicts are each feature name in each subset with its corresponding index in the dataframe.
	  '''
	feature_dicts = [{pddf_features.get_loc(feature): \
					  feature for feature in sublist} \
					  for sublist in fnsub]
	return feature_dicts

#this function returns the set of all names that are being requested.
#these are feature names that will likely be used to drop from the used dataset
def return_name_collection():

	set1 = fn_hilo_prices(0)
	set2 = fn_ma_prices(0)
	set3 = fn_orig_price(0)

	full_set = set1+set2+set3

	return full_set

def fn_vel(index):
	return [f'vel{i}_{idx[index]}' for i in range(1,61)]
def fn_acc(index):
	return [f'acc{i+1}_{idx[index]}' for i in range(60)]
def fn_stoch_k(index):
	return [f'stchK{i}_{idx[index]}' for i in range(5, 125, 5)]
def fn_vol_m(index):
	'''NOTE subset 1 of fe_vol_sz_diff feature set END NOTE'''
	return [f'vol_m{i}_{idx[index]}' for i in range(2,61)]
def fn_vol_avgDiff(index):
	'''NOTE subset 2 of fe_vol_sz_diff feature set END NOTE'''
	return [f'vol_avgDiff{i}_{idx[index]}' for i in range(2,61)]
def fn_ma_s60(index):
	'''NOTE the sub 60m subset of ma feature set'''
	return [f'ma{i}_{idx[index]}' for i in range(2,60)]
def fn_ma_s240(index):
	'''NOTE the sub 240m subset of ma feature set'''
	return [f'ma{i}_{idx[index]}' for i in range(60,241,20)]
def fn_disp_ma60(index):
	'''NOTE the sub 60m subset of ma feature set'''
	return [f'disp_ma{i}_{idx[index]}' for i in range(2,60)]
def fn_disp_ma240(index):
	'''NOTE the sub 240m subset of ma feature set'''
	return [f'disp_ma{i}_{idx[index]}' for i in range(60,241,20)]
def fn_ma_diff(index):
	cols = []
	lengths = [5,15,30,60,120,240]
	for i in range(len(lengths)):
		for j in range(i+1,len(lengths)):
			cols.append(f'diff_ma_{lengths[i]}_{lengths[j]}_{idx[index]}')
	return cols

'''NOTE TIMES-------------REFERENCE NOTE'''
#NOTE times = [5,15,30,60,120,240]''' NOTE#
'''NOTE TIMES-------------REFERENCE NOTE'''

def fn_hihi(index):
	'''NOTE first subset of fe_hihi_diff END NOTE'''
	return [f'hihi{i}_{idx[index]}' for i in times]
def fn_disp_hihi(index):
	'''NOTE second subset of fe_hihi_diff END NOTE'''
	return [f'disp_hihi{i}_{idx[index]}' for i in times]
def fn_lolo(index):
	'''NOTE first subset of fe_lolo_diff END NOTE'''
	return [f'lolo{i}_{idx[index]}' for i in times]
def fn_disp_lolo(index):
	'''NOTE second subset of fe_lolo_diff END NOTE'''
	return [f'disp_lolo{i}_{idx[index]}' for i in times]
def fn_hilo_stoch(index):
	cols = []
	#prepping the feature names according to ma's used
	for i in range(len(times)):
		for j in range(len(times)):
			cols.append(f'hilo_stoch_{times[i]}_{times[j]}_{idx[index]}')
	return cols

def fn_hilo_diff(index):
	lengths = [5,15,30,60,120,240]
	cols = []
	#prepping the feature names according to ma's used
	for i in range(len(lengths)):
		for j in range(len(lengths)):
			cols.append(f'diff_hilo_{lengths[i]}_{lengths[j]}_{idx[index]}')
	return cols

def fn_orig_price(index=None):
	if(index == None):
		return ['high','low','close','high.1','low.1','close.1']
	elif(index == 0):
		return ['high','low','close']
	elif(index == 1):
		return ['high.1','low.1','close.1']
	else:
		raise ValueError(f"Fatal: fn_orig_price <-- return_name_collection <-- _Feature_Usage:\nIndex value of {index} not interpretable.")

def fn_orig_vol():
	return ['volume','volume.1']

def fn_orig_time():
	return ['time']

def fn_hilo_prices(index):
	fn_hihi = [f'hihi{i}_{idx[index]}' for i in times]
	fn_lolo = [f'lolo{i}_{idx[index]}' for i in times]
	cols = fn_hihi+fn_lolo
	return cols

def fn_ma_prices(index):
	cols = [f'ma{i}_{idx[index]}' for i in range(2,60)]+\
		   [f'ma{i}_{idx[index]}' for i in range(60,241,20)]
	return cols