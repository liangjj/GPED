from GPED.BasisInfo import BasisInfo
from GPED.OperatorMat import OperatorMat
from GPED.getMat import __MultOp
from GPED.BasisSet import BasisSet, default_check_succeed
import scipy.sparse as sparse

import numpy as np
from multiprocessing import Pool
from functools import partial

#Translation operator
class TOp:
    def __init__(self, basisinfo, q=1, isfermion = False, phi = 0, Ne = 0):
        self.offset = basisinfo.numBitPerSite
        self.N = basisinfo.Nsite
        self.q = q
        self.isfermion = isfermion
        self.phi = phi
        self.Ne = Ne
        
        if isfermion and Ne == 0:
            raise RunTimeError
        
        if(self.N%q != 0):
            raise RunTimeError
            
    def __getitem__(self, ket):
        if type(ket) is not tuple:
            n = 1
        else:
            n = ket[1]
            ket = ket[0]
            
        coeff = 1
        Tket = ket
        while n > 0:
            c, Tket = self.__applyT(Tket)
            coeff *= c
            n = n - 1
        return coeff, Tket
        
    def __applyT(self, ket):
        Tket = ket
        
        coeff = 1.0
        for i in range(self.q*self.offset):
            res = Tket % ( 1 << 1 )
            Tket = (Tket >> 1) + (res << self.N*self.offset-1)
            
            if res == 1:
                coeff = coeff * np.exp(1j*self.phi)
                if self.isfermion:
                    coeff = coeff * (-1)**(self.Ne-1)
        
        if(abs(np.imag(coeff)) <1E-10):
            coeff = np.real(coeff)
        return coeff, Tket
    
#find the representation state
class repre:
    def __init__(self, T):
        self.T = T
        
    def __getitem__(self, ket):
        MIN = ket
        for i in range(1, self.T.N):
            coeff, ket = self.T[ket]
            if(ket < MIN):
                MIN = ket
        return MIN
    
# <bra|P_k|ket>
# P_k = q/N*sum_j exp(-2*pi*i*j*k*q/N)*(T_q)^j
class Pexp:
    def __init__(self, T, k):
        self.k = k
        self.T = T
        
    def __getitem__(self, braket):
        bra = braket[0]
        ket = braket[1]
        coeff = 0
        
        for j in range(int(self.T.N/self.T.q)):
            c, Tjket = self.T[ket, j]
            if bra == Tjket:
                coeff = coeff + c*np.exp(2*np.pi*1j*j*self.k*self.T.q/self.T.N)*self.T.q/self.T.N
        return coeff
    
    
class BasisSetTS(BasisSet):
    
    def __init__(self, basisinfo, qn, k, T = [],
                check_succeed = default_check_succeed, BSet = None):
        
        if BSet is not None:
            BasisSet.__init__(self, basisinfo, qn, check_succeed, store = BSet.store)
        else:
            BasisSet.__init__(self, basisinfo, qn, check_succeed)
            
        self.storePcoeff = dict()
        
        self.k = k
        
        
        # helper class to perfrom Translation operator
        # T[ket] = T|ket>
        if not T:
            self.T = TOp(basisinfo)
        else:
            self.T = T
        
        self.q = self.T.q
        
        # helper class to find the representative state
        # |r> = R[ket]
        self.R = repre(self.T)
        
        # helper class to calculate <bra|P_k|ket>
        # <bra|P_k|ket> = Pexp[bra, ket]
        self.Pexp = Pexp(self.T, k)
        
        # find the representative state and <ket|P_k|ket>
        repreKets = dict()
        for b in self.store:
            r = self.R[b]
            if(r not in repreKets):
                p = self.Pexp[r,r]
                if(abs(p)>1E-8):
                    self.storePcoeff[r] = np.real(p)
                    repreKets[r] = len(repreKets)
        
        # delete the original basis
        del self.store
        
        # replace the orginal BasisSet with representative kets
        self.store = repreKets
    
    # return <ket|P_k|ket>
    def getCoeff(self, ket):
        return self.storePcoeff[ket]


def __getMatTS_helper(ket, opCoeffSet, N, k, q, opinfo, BSet, multop):
    M_bra_coeff = dict()
            
    # <r|P_k|r>
    r_Pk_r = BSet.getCoeff(ket)
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
        # P_k (H|ket>)
        #######
        if(abs(coeff) > 1E-8):
            Hket = bra
            # apply P_k to (H|ket>)
            # P_k = q/N*sum_{j=0}^(N/q-1) exp(-2*pi*i*j*k*q/N)*T^j
            #
            # <r_k'|H|r_k> = <r'|P_k*H|r>/sqrt(<r'|P_k|r'><r|P_k|r>)
            #   = sum_{jj=0}^(N/q-1) [q/(N*sqrt(prr'*prr))*exp(-2*pi*i*jj*k*q/N)] * <r'|T^jj*H|r>
            #
            # ,where prr = <r|P_k|r>, prr' = <r'|P_k|r'>
            for jj in range(int(N/q)):
                if(Hket in BSet):
                    # <r'|P_k|r'>
                    rp_Pk_rp = BSet.getCoeff(Hket)
                    
                    #[q/(N*sqrt(prr'*prr))*exp(-2*pi*i*jj*k*q/N)]
                    p = q/(N*np.sqrt(r_Pk_r*rp_Pk_rp+0j))*np.exp(-2.0*np.pi*1j*jj*k*q/N)
                    
                    if Hket in M_bra_coeff:
                        M_bra_coeff[Hket] += coeff*p
                    else :
                        M_bra_coeff[Hket] = coeff*p
                            
                c, Hket = BSet.T[Hket]
                coeff = coeff*c
                
    return [ket, M_bra_coeff]
    
def getMatTS(OpMat, BSet):
    #<r_k'|H|r_k> = <r'|P_k*H|r>/sqrt(<r'|P_k|r'><r|P_k|r>)
    #P_k = 1/N*sum_j exp(-2*pi*i*j*k/N)*T^j
    
    N = OpMat.basisinfo.Nsite
    
    opinfo = OpMat.operatorinfo
    multop = __MultOp(opinfo)
    
    L = [] 
    '''
    for ket in BSet.getBasis():
        L.append(__getMatTS_helper(ket, OpMat.opCoeffSet, N, BSet.k, BSet.q, OpMat.operatorinfo, BSet, multop))
    '''
    with Pool() as pool:
        L = pool.map(partial(__getMatTS_helper, opCoeffSet = OpMat.opCoeffSet, N = N, k = BSet.k, q = BSet.q, 
                                             opinfo = OpMat.operatorinfo, multop = multop, BSet = BSet), BSet.getBasis() )
    D = len(BSet)
    O = sparse.lil_matrix((D,D), dtype = complex)
    for l in L:
        i = BSet[l[0]]
        for bra in l[1].keys():
            j = BSet[bra]
            O[j,i] = l[1][bra]
    return O
