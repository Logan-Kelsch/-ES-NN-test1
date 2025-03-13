'''
'''

import operator

class Gene():
    '''
    '''
    def __init__(
        self,
        patterns    :   list    =   []
    ):
        
        self._patterns = patterns
        return
    
    #patterns - list of classes
    #index range or index list

    @property
    def patterns(self):
        return self._patterns
    
    @patterns.setter
    def patterns(self, new:any):
        self._patterns = new



class Pattern():
    '''
    '''
    def __init__(
        v1  :   any =   None,
        l1  :   any =   None,
        op  :   any =   None,
        v2  :   any =   None,
        l2  :   any =   None
    ):
        
        return
    #var1 - feature index
    #operator - (<, >)
    #var2 - feature index

    #generate random
    def random(
        acceptable_vals :   list    =   [],
        acceptable_lags :   list|range  =   None
    ):
        assert acceptable_lags != None, \
            f"A random pattern was requested, but {acceptable_lags} acceptable lags were entered."



    @property
    def v1(self):
        return self._v1
    
    @property
    def op(self):
        return self._op
    
    @property
    def v2(self):
        return self._v2
    
    @v1.setter
    def v1(self, new:any):
        self._v1 = new

    @op.setter
    def op(self, new:any):
        self._op = new

    @v2.setter
    def v2(self, new:any):
        self._v2 = new
