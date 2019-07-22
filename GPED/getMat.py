from GPED.BasisInfo import BasisInfo
from GPED.OperatorInfo import OperatorInfo
from GPED.OperatorMat import OperatorMat

import scipy.sparse as sparse
from multiprocessing import Pool
from functools import partial

    
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


def getMat(OpMat, BSet, multiprocessing = True):
    
    if(multiprocessing):
        with Pool() as pool:
            L = pool.map(partial(__OpMat2matele_helper, opCoeffSet = OpMat.opCoeffSet, 
                                  opinfo = OpMat.operatorinfo), BSet.getBasis())
    else:
        L = []
        for ket in BSet.getBasis():
            L.append(__OpMat2matele_helper(ket, OpMat.opCoeffSet, OpMat.operatorinfo))
            
    D = len(BSet)
    
    O = sparse.lil_matrix((D, D), dtype = OpMat.dtype)
    for l in L:
        i = BSet[l[0]]
        for bra in l[1].keys():
            j = BSet[bra]
            O[j,i] = l[1][bra]
            
    return O