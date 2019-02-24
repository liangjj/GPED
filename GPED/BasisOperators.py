from GPED.BasisInfo import BasisInfo
from GPED.OperatorInfo import OperatorInfo
import numpy as np

def SpinlessFermion(N):
    Basis = BasisInfo(['0','1'], N)
    for i in range(N):
        Basis.put(i, '0', 0)
        Basis.put(i, '1', 1)
    Operator = OperatorInfo(Basis)
    for i in range(N):
        Operator['n',i] = [[0, 0]
                          ,[0, 1]]
        Operator['C',i] = [[0, 1]
                          ,[0, 0]]
        Operator['Cdag',i] = [[0, 0]
                             ,[1, 0]]
        
        Operator['A',i] = [[0, 1]
                          ,[0, 0]]
        Operator['Adag',i] = [[0, 0]
                             ,[1, 0]]
    
    return Basis, Operator
                             

def SpinfulFermion(N):
    BInfo = BasisInfo( ['0','u', 'd', '2'] , N)
    for i in range(N):
        BInfo.put(i, '0', [0, 0])
        BInfo.put(i, 'u', [1, 1])
        BInfo.put(i, 'd', [1,-1])
        BInfo.put(i, '2', [2, 0])
    
    OpInfo = OperatorInfo(BInfo)
    for i in range(N):
        OpInfo['Cup',i] = [[0, 1, 0, 0]
                          ,[0, 0, 0, 0]
                          ,[0, 0, 0, 1]
                          ,[0, 0, 0, 0]]
    
        OpInfo['Cdagup',i] = [[0, 0, 0, 0]
                             ,[1, 0, 0, 0]
                             ,[0, 0, 0, 0]
                             ,[0, 0, 1, 0]]
    
        OpInfo['Cdn',i] = [[0, 0, 1, 0]
                          ,[0, 0, 0, 1]
                          ,[0, 0, 0, 0]
                          ,[0, 0, 0, 0]]
    
        OpInfo['Cdagdn',i] = [[0, 0, 0, 0]
                             ,[0, 0, 0, 0]
                             ,[1, 0, 0, 0]
                             ,[0, 1, 0, 0]]
        
        OpInfo['nup',i] = [[0, 0, 0, 0]
                          ,[0, 1, 0, 0]
                          ,[0, 0, 0, 0]
                          ,[0, 0, 0, 1]]
        
        OpInfo['ndn',i] = [[0, 0, 0, 0]
                          ,[0, 0, 0, 0]
                          ,[0, 0, 1, 0]
                          ,[0, 0, 0, 1]]
        
    return BInfo, OpInfo


def SpinHalf(N):
    Basis = BasisInfo(['u','d'], N)
    for i in range(N):
        Basis.put(i, 'u', 0.5)
        Basis.put(i, 'd', -0.5)
    Operator = OperatorInfo(Basis)
    for i in range(N):
        Operator['Sz',i] = [[1, 0]
                          , [0, -1]]
        
        Operator['Sx',i] = [[0, 1]
                           ,[1, 0]]
        
        Operator['Sy',i] = [[0, -1j]
                           ,[1j, 0]]
        
        Operator['S+',i] = [[0,  1]
                           ,[0,  0]]
        
        Operator['S-',i] = [[0,  0]
                           ,[1,  0]]
    
    return Basis, Operator


def HardCoreBoson(N):
    Basis = BasisInfo(['0','1'], N)
    for i in range(N):
        Basis.put(i, '0', 0)
        Basis.put(i, '1', 1)
    Operator = OperatorInfo(Basis)
    for i in range(N):
        Operator['n',i] = [[0, 0]
                          ,[0, 1]]
        Operator['A',i] = [[0, 1]
                          ,[0, 0]]
        Operator['Adag',i] = [[0, 0]
                             ,[1, 0]]
    
    return Basis, Operator


def Boson(N, nb):
    BInfo = BasisInfo([str(i) for i in range(nb+1)], N)
    for i in range(N):
        for s in range(nb+1):
            BInfo.put(i, str(s), s)
    OpInfo = OperatorInfo(BInfo)
    for i in range(N):
        for n in range(nb):
            OpInfo.append('Adag', i, str(n), str(n+1), np.sqrt(n+1))
            OpInfo.append('A', i, str(n+1), str(n), np.sqrt(n+1))
        for n in range(1, nb+1):
            OpInfo.append('n', i, str(n), str(n), n)
            OpInfo.append('n-1', i, str(n), str(n), n-1)
            
    return BInfo, OpInfo