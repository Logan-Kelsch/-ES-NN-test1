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