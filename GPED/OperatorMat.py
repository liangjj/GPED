from GPED.BasisInfo import BasisInfo
from GPED.OperatorInfo import OperatorInfo
import numpy as np

class OperatorMat:
    def __init__(self, operatorinfo):
        
        self.basisinfo = operatorinfo.basisinfo
        self.operatorinfo = operatorinfo
        
        self.opCoeffSet = []
        self.dtype = float
        
    '''
    OpCoeffList: a list, OpCoeffList[0] is the coefficient
                and the rest of the elements in the list are 
                Opname, position pairs.
                i.e. (t b_i^\dagger b_{i+1}^-) : [t, "bdag", i, "b", i+1]
    '''
    def __add__(self, OpCoeffList):
        if(abs(OpCoeffList[0]) > 1E-10):
            if(type(OpCoeffList[0]) == complex or type(OpCoeffList[0]) == np.complex128):
                self.dtype = complex
            self.opCoeffSet.append(OpCoeffList)
        return self
    
    def __str__(self):
        return str(self.opCoeffSet)