import numpy as np
from GPED.BasisInfo import BasisInfo

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


