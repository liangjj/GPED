import numpy as np
import scipy.sparse as sparse
from multiprocessing import Pool
from functools import partial
import time
from Basis import BasisInfo

'''
class OperatorInfo

Constructor(BasisInfo)

__setitem__([opname, position], matrix)

__getitem__([opname, position])

append(opname, position, fromstate, tostate, coeff)

'''

'''
class OperatorMat

Constructor(BasisInfo, OperatorInfo, BasisSet)

__add__([coeff, O1, p1, O2, p2, ...., ON, pN])

'''

'''

'''
class OperatorInfo:
    def __init__(self, basisinfo):
        self.basisinfo = basisinfo
        
        '''
        How to store the operators?
        
        Suppose that we have an operator O_p = \sum_{a,b} C_{ab}|a><b|_p
        , where p is the position index of the operator 
        and C_{ab} is the coefficient of the operator
        
        The operator will be store in the following dictionary:
        
        key : bin(|b>_p), state |b> in binary representation
        
        value : [bin(|a>_p), C_{ab}, p]
        ,where bin(|a>_p) is state |a> in binary representation.
        
        This storage enables efficient calculation of O|ket> 
        i.e. to calculate the operation |ket'> = O|ket> 
            bin(|ket'>) = (bin(|ket>) + bin(|a>_p) - bin(|b>_p))
        '''
        self.store = dict()
        
    def __setitem__(self, opnamePos, mat):
        opname = opnamePos[0]
        pos = opnamePos[1]
        
        if len(mat) != self.basisinfo[pos].dim():
            print(self.basisinfo[pos].dim())
            raise RuntimeError
        if len(mat) != len(mat[0]):
            raise RuntimeError
        
        MAT = dict()
        i = 0
        for ket in self.basisinfo.getState():
            bket = i << self.basisinfo.numBitPerSite*pos
            j = 0
            for bra in self.basisinfo.getState():
                if(abs(mat[j][i])>1E-8):
                    bbra = j << self.basisinfo.numBitPerSite*pos
                    MAT[bket] = [bbra, mat[j][i], pos]
                j = j + 1
            i = i + 1
        self.store[opname+'_'+str(pos)] = MAT
    
    def append(self, opname, position, fromstate, tostate, coeff):
        
        if(abs(coeff) < 1E-8):
            return
        
        i = self.basisinfo.stateSet.index(fromstate)
        j = self.basisinfo.stateSet.index(tostate)
        bket = i << self.basisinfo.numBitPerSite*position
        bbra = j << self.basisinfo.numBitPerSite*position
        
        key = opname+'_'+str(position)
        if not(key in self.store):
            self.store[key] = dict()
        self.store[key][bket] = [bbra, coeff, position]

    def __getitem__(self,opnamePos):
        opname = opnamePos[0]
        pos = opnamePos[1]
        return self.store[opname+'_'+str(pos)]
    
    def __str__(self):
        s = 'Operator Info\n\n'
        
        for opname in self.store.keys():
            s = s + opname + ' = '
            remain = len(self.store[opname])
            count = 0
            for bket in self.store[opname].keys():
                bbra, coeff, pos = self.store[opname][bket]
                fromstate = self.basisinfo.stateSet[bket>>self.basisinfo.numBitPerSite*pos]
                tostate = self.basisinfo.stateSet[bbra>>self.basisinfo.numBitPerSite*pos]
                
                if(abs(coeff%1)<1E-8):
                    if((coeff-1)>1E-8):
                        s = s + '%d'%coeff 
                else:
                    s = s + '%2.2f'%coeff 
                    
                s = s + '|' + tostate + '>' + '<' + fromstate + '|' + '_' + str(pos)
                remain = remain-1
                count = count +1
                if(count % 5 == 0):
                    s = s + '\n\t'
                if(remain > 0):
                    s = s + ' + '
                else:
                    s = s + '\n'
            s = s + '\n'
        return s

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

    
'''
Helper class to perfrom operation on state
i.e. to calculate |ket'> = O_i|ket>
M = __MultOp(OpInfo)
ket' = M[O,i,ket]
'''
class __MultOp:
    def __init__(self, operatorinfo):
        self.operatorinfo = operatorinfo
        self.basisinfo = operatorinfo.basisinfo
    
    def __getitem__(self, OpnamePosKet):
        opname = OpnamePosKet[0]
        pos = OpnamePosKet[1]
        ket = OpnamePosKet[2]
        
        # operator
        Op = self.operatorinfo[opname, pos]
        
        #from state
        fromState = self.basisinfo.getBinatPosition(pos, ket)
        if(not(fromState in Op)):
            return [ket, 0]
        
        #to state 
        toState, coeff, pos= Op[fromState]
        
        # get Oket
        delta = -fromState + toState
        Oket = ket + delta
        
        # Jordan-Wigner transformation
        # here we use a slightly different convension to make implementation easier:
        # \sigma_i = \prod_{j = i+1}^N (-1)^(n_j) C_i
        if(opname[0] == 'C'):
            temOket = Oket
            fermiParity = 0
            reference = abs(delta)
            while(reference%2 == 0):
                fermiParity = fermiParity + temOket%2
                temOket = temOket >> 1
                reference = reference >> 1
            coeff = coeff * (-1)**fermiParity
        return Oket, coeff

    
# calculate the matrix element <bra|M|ket> and store it in a dict {bra: <bra|M|ket>}
def __OpMat2matele_helper(ket, opCoeffSet, opinfo):
    
    multop = __MultOp(opinfo)
    M_bra_coeff = dict()
    
    # Apply Operators 
    for OpCoeffList in opCoeffSet:
        ##############################################################
        # make sure O1*O2*....*ON|ket> != 0 (it is zero in most cases)
        ######
        stop = False
        index = len(OpCoeffList)-1
        #if two or more operators are in the same positions
        #, only check the first operator 
        checked_pos = set()
        while index > 0:
            pos = OpCoeffList[index]
            if(pos in checked_pos):
                index = index - 2
                continue
            opname = OpCoeffList[index-1]
            Op = opinfo[opname, pos]
            fromState = opinfo.basisinfo.getBinatPosition(pos, ket)
            if(not(fromState in Op)):
                stop = True
                break
            index = index - 2
            checked_pos.add(pos)
        if(stop):
            continue
        ##############################################################
        # apply a series of operators
        #######
        coeff = OpCoeffList[0]
        bra = ket
        # apply operators successively
        index = len(OpCoeffList)-1
        while index > 0 and abs(coeff) > 1E-8 :
            position = OpCoeffList[index]
            opname = OpCoeffList[index-1]
            bra, c = multop[opname, position, bra]
            coeff = coeff*c
            index = index - 2
        ##############################################################
        # fill in the matrix element
        #########
        if(abs(coeff) > 1E-8):
            if bra in M_bra_coeff:
                M_bra_coeff[bra] = M_bra_coeff[bra] + coeff
            else :
                M_bra_coeff[bra] = coeff
                
    return [ket, M_bra_coeff]


def getMat(OpMat, BSet):
    
    with Pool() as pool:
        L = pool.map(partial(__OpMat2matele_helper, opCoeffSet = OpMat.opCoeffSet, 
                                  opinfo = OpMat.operatorinfo), BSet.getBasis())
        
    D = len(BSet)
    
    O = sparse.lil_matrix((D, D), dtype = OpMat.dtype)
    for l in L:
        i = BSet[l[0]]
        for bra in l[1].keys():
            j = BSet[bra]
            O[j,i] = l[1][bra]
            
    return O
