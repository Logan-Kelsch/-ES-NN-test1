'''
'''

import operator
import random
from typing import Literal
import numpy as np
import _09_utility as util

class Gene():
	'''
	'''
	def __init__(
		self,
		patterns    :   list    =   []
	):
		
		self._patterns = patterns

		self._lastarray_presence = None
		self._lastarray_returns = None
		self._lastarray_kelsch_ratio = None

		self._lastavg_returns = None
		self._lastavg_kelsch_ratio = None
		self._last_profit_factor = None

		return	
	
	def copy(
		self,
		patterns_only	:	bool	=	True
	):
		#make empty gene
		new = Gene()

		#copy over patterns
		new._patterns = self._patterns

		#if desired, copy over everything else
		if(not patterns_only):
			new._lastarray_presence = self._lastarray_presence
			new._lastarray_returns = self._lastarray_returns
			new._lastarray_kelsch_ratio = self._lastarray_kelsch_ratio
			new._lastavg_returns = self._lastavg_returns
			new._lastavg_kelsch_ratio = self._lastavg_kelsch_ratio
			new._last_profit_factor = self._last_profit_factor

		#return duplicate gene
		return new


	def show_patterns(
		self,
		fss
	):
		
		patterns = ""
		for pattern in self._patterns:
			patterns += pattern.show(fss)

		return patterns

	def custom(
		self,
		fss				:	list,
		acceptable_lag	:	list,
		pattern_vals	:	list
	):
		'''
		This function takes a list of tuples|lists pattern values and loads them into a gene
		'''

		#need to ensure that all pattern_vals coming in are of logical structure
		for i, p in enumerate(pattern_vals):
			assert len(p) == 5, f"Custom pattern #{i+1} came in with {len(p)} parameters, need exactly 5."
		
		self._pattern = []

		#for ease of declaring operator from string
		op_map = {
			"lt" : operator.lt,
			"gt" : operator.gt
		}

		#for each new pattern, add according to provided data
		for newp in pattern_vals:
			
			self._patterns.append(
				Pattern(
					#take on fss collection overhead here to avoid needing to find and add by hand
					acceptable_vals=util.get_fss_from_value(fss,newp[0]),
					#pass in lags and all other params
					acceptable_lags=acceptable_lag,
					v1=newp[0],
					l1=newp[1],
					op=op_map[newp[2]],
					v2=newp[3],
					l2=newp[4]
				)
			)
		

	#this funcction takes in arrays and performances to save locally to gene class
	def update(
		self,
		lastarray_returns	:	any	=	None,
		lastarray_kelsch_ratio:	any	=	None,
		lastavg_returns		:	any	=	None,
		lastavg_kelsch_ratio:	any	=	None,
		last_profit_factor	:	any	=	None
	):
		'''
		### info:
		This function takes in arrays and performances to save locally to gene class
		'''
		
		#if(lastarray_returns != None):
		self._lastarray_returns = lastarray_returns
		
		#if(lastarray_kelsch_ratio != None):
		self._lastarray_kelsch_ratio = lastarray_kelsch_ratio
		
		#if(lastavg_returns != None):
		self._lastavg_returns = lastavg_returns
		
		#if(lastavg_kelsch_ratio != None):
		self._lastavg_kelsch_ratio = lastavg_kelsch_ratio

		#if(last_profit_factor != None):
		self._last_profit_factor = last_profit_factor


	#patterns - list of classes
	#index range or index list

	@property
	def patterns(self):
		return self._patterns
	
	@patterns.setter
	def patterns(self, new:any):
		self._patterns = new

	#arrays for holding onto returns and ratio values for evaluation

	@property
	def lastarray_returns(self):
		return self._lastarray_returns
	
	@lastarray_returns.setter
	def lastarray_returns(self, new:any):
		self._lastarray_returns = new

	@property
	def lastarray_kelsch_ratio(self):
		return self._lastarray_kelsch_ratio
	
	@lastarray_kelsch_ratio.setter
	def lastarray_kelsch_ratio(self, new:any):
		self._lastarray_kelsch_ratio = new

	#individual values used for gene evaluation

	@property
	def lastavg_returns(self):
		return self._lastavg_returns
	
	@lastavg_returns.setter
	def lastavg_returns(self, new:any):
		self._lastavg_returns = new

	@property
	def lastavg_kelsch_ratio(self):
		return self._lastavg_kelsch_ratio
	
	@lastavg_kelsch_ratio.setter
	def lastavg_kelsch_ratio(self, new:any):
		self._lastavg_kelsch_ratio = new

	@property
	def last_profit_factor(self):
		return self._last_profit_factor
	
	@last_profit_factor.setter
	def last_profit_factor(self, new:any):
		self._last_profit_factor = new


class Pattern():
	'''
	### info: ###

	The pattern class is used for comparing two values at two locations in a dataset, in respect to a local index value.

	### params: ###

	- v1 / v2:
	-	-	integers containing feature index values
	- l1 / l2:
	-	-	integers containing lag (sample displacement) values
	- op:
	-	-	operator for comparison between values

	### example: ###

	feature v1 at (sample - l1) op feature v2 at (sample - l2) <br>
	>>> #returns boolean
	>>> ptrn_istrue = ptrn._op(data[(curr_smpl-ptrn._l1) , ptrn._v1], data[(curr_smpl-ptrn._l2) , ptrn._v2])
	'''

	def __init__(
		self,
		acceptable_vals :   list    =   [],
		acceptable_lags :   list|range    =   [],
		v1  :   int =   None,
		l1  :   int =   None,
		op  :   operator	=   None,
		v2  :   int =   None,
		l2  :   int =   None
	):
		assert acceptable_lags != None, \
			f"A pattern was created, but {acceptable_lags} acceptable lags were entered."
		assert acceptable_vals != None, \
			f"A pattern was created, but {acceptable_vals} acceptable vals were entered."
		
		self._acceptable_lags = acceptable_lags
		self._acceptable_vals = acceptable_vals

		#var1 - feature index
		#lag1 - lag offset
		#operator - (< or >) only
		#var2 - feature index
		#lag2 - lag offset
		self._v1 = v1
		self._l1 = l1
		self._op = op
		self._v2 = v2
		self._l2 = l2

		#randomization for initial creation (only acceptable lags and vals are entered)
		#if any v1,v2,l1,l1,op are None, this means these are initial pattern creations, therefore..
		missing = self.has_missing()
		for miss in missing:
			self.switch(miss)
		
		return
	
	def show(
		self,
		fss
	):
		
		fd = util.get_full_feature_dict(fss)
		
		op_map = {
			operator.lt : "<",
			operator.gt : ">"
		}
		
		return f"({self._v1}){fd[self._v1]}[{self._l1}] {op_map[self._op]} ({self._v2}){fd[self._v2]}[{self._l2}]\n"
	
	def has_missing(
		self
	):
		missing = []
		if(self._v1 == None):
			missing.append('v1')
		if(self._v2 == None):
			missing.append('v2')
		if(self._l1 == None):
			missing.append('l1')
		if(self._l2 == None):
			missing.append('l2')
		if(self._op == None):
			missing.append('op')
		
		return missing
		
	
	def equals(
		self,
		foreign	:	any	=	None
	):
		#if is an illegal format
		if(not isinstance(foreign, Pattern)):
			return False

		#now check each parameter to see if they are identical
		
		if(self._v1 != foreign._v1):
			return False
		
		if(self._l1 != foreign._l1):
			return False
		
		if(self._op != foreign._op):
			return False
		
		if(self._v2 != foreign._v2):
			return False
		
		if(self._l2 != foreign._l2):
			return False
		
		#all parameters are identical if this area is reached
		return True
		
		

	#generate random
	def random(
		self,
		mutation	:	bool	=	False,
		fss			:	list	=	None
	):
		
		#check if this is a mutation call, if so, replace the fss (possibly)
		if(mutation):
			self.switch(type='fs', fss=fss)
		#This is successfull on initial call, 
		#as switch has full value inclusion when called on None values
		self._v1 = self.switch('v1')
		self._v2 = self.switch('v2')
		self._l1 = self.switch('l1')
		self._l2 = self.switch('l2')
		self._op = self.switch('op')
		
	def switch(
		self,
		type	:	Literal['op','l1','v1','l2','v2','fs','mutation']	=	None,
		spec	:	any	=	None,
		fss		:	list=	None
	):
		'''
		### info: ###
		This function takes a given pattern parameter and replaced with a specified value, or a random value in none is specified.
		
		### params: ###
		- type:
		-	-	choose the parameter that will be switched.
		- spec:
		-	-	optional specific value to switch to, will select (NEW/DIFFERENT) acceptable value if not specified. 
		'''
		assert type != None, \
			"pattern.switch was requested, but switch type was defined as 'None'."
		
		#if the sample coming in is a mutation case, pick random new switcher
		if(type == 'mutation'):
			#pick random value to switch
			type = random.choice(['op','l1','v1','l2','v2'])
			#enforce this comes in as None so that it can be randomly generated below
			spec = None

		
		#if spec comes in as None, that means a random value is requested
		#for each case, if random is requested, ensure we are not selecting
		#the same value a second time.
		if(spec == None):
			match(type):
				case 'op':
					tmp = [operator.lt,operator.gt]
					self._op = random.choice([o for o in tmp if tmp != self._op])
				case 'l1':
					self._l1 = random.choice([n for n in self._acceptable_lags if n != self._l1])
				case 'l2':
					self._l2 = random.choice([n for n in self._acceptable_lags if n != self._l2])
				case 'v1':
					self._v1 = random.choice([n for n in self._acceptable_vals if n != self._v1])
				case 'v2':
					self._v2 = random.choice([n for n in self._acceptable_vals if n != self._v2])
				case 'fs':
					self._acceptable_vals = random.choice(fss)
				case _:
					raise ValueError("FATAL: bad 'type' value in some_pattern.switch function call. (Random Switch)")
		#this case is reached when we are entering a specified value
		else:
			match(type):
				case 'op':
					self._op = spec
				case 'l1':
					self._l1 = spec
				case 'l2':
					self._l2 = spec
				case 'v1':
					self._v1 = spec
				case 'v2':
					self._v2 = spec
				case 'fs':
					self._acceptable_vals = fss[spec]
				case _:
					raise ValueError("FATAL: bad 'type' value in some_pattern.switch function call. (Specific Switch)")


	#universal feature subsets, passed in and does not chance
								
	#acceptable lags and vals list/ranges for logical and repeatable operating. properties and setters

	@property
	def acceptable_vals(self):
		return self._acceptable_vals
	
	@acceptable_vals.setter
	def acceptable_vals(self, new:any):
		self._acceptable_vals = new

	@property
	def acceptable_lags(self):
		return self._acceptable_lags
	
	@acceptable_lags.setter
	def acceptable_lags(self, new:any):
		self._acceptable_lags = new

	#variable, lag, and operation properties and setters

	@property
	def v1(self):
		return self._v1
	
	@property
	def l1(self):
		return self._l1
	
	@property
	def op(self):
		return self._op
	
	@property
	def v2(self):
		return self._v2
	
	@property
	def l2(self):
		return self._l2
	
	@v1.setter
	def v1(self, new:any):
		self._v1 = new

	@l1.setter
	def l1(self, new:any):
		self._l1 = new

	@op.setter
	def op(self, new:any):
		self._op = new

	@v2.setter
	def v2(self, new:any):
		self._v2 = new

	@l2. setter
	def l2(self, new:any):
		self._l2 = new
