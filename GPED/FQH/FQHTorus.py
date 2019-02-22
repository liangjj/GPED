from Basis import BasisInfo
from Operator import OperatorInfo

def torus_check_succeed(qn_curr, qn_goal, basisinfo):
    if qn_curr[0] != qn_goal[0]:
        return False
    
    if qn_curr[1]%basisinfo.Nsite != qn_goal[1]%basisinfo.Nsite:
        return False
    
    if(len(qn_curr)>2):
        for i in range(2, len(qn_curr)):
            if(qn_curr[i] != qn_goal[i]):
                return False
    return True

def FQHTorus(N):
    
    BInfo = BasisInfo(['0','1'], N)
    for i in range(N):
        BInfo.put(i, '0', [0, 0])
        BInfo.put(i, '1', [1, i])
        
    OpInfo = OperatorInfo(BInfo)
    
    for i in range(N):
        OpInfo['n',i] = [[0, 0]
                        ,[0, 1]]
    
        OpInfo['C',i] = [[0, 1]
                        ,[0, 0]]
    
        OpInfo['Cdag',i] = [[0, 0]
                           ,[1, 0]]
        
        OpInfo['A',i] = [[0, 1]
                        ,[0, 0]]
    
        OpInfo['Adag',i] = [[0, 0]
                           ,[1, 0]]
    return BInfo, OpInfo



def FQHTorus_bilayer(N, conserveSz = True):
    
    # state of a site
    # 0 : vac
    # u : 1 electron in upper layer
    # d : 1 electron in lower layer
    # 2 : 2 electrons in both layers
    
    statelist = ['0','u', 'd', '2']
    
    BInfo = BasisInfo(statelist, N)
    if(conserveSz):
        for i in range(N):
            BInfo.put(i, '0', [0,   0, 0])
            BInfo.put(i, 'u', [1,   i, 1])
            BInfo.put(i, 'd', [1,   i,-1])
            BInfo.put(i, '2', [2, 2*i, 0])
    else:
        for i in range(N):
            BInfo.put(i, '0', [0,   0])
            BInfo.put(i, 'u', [1,   i])
            BInfo.put(i, 'd', [1,   i])
            BInfo.put(i, '2', [2, 2*i])
            
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
        
        OpInfo['Nup', i] = [[0, 0, 0, 0]
                           ,[0, 1, 0, 0]
                           ,[0, 0, 0, 0]
                           ,[0, 0, 0, 1]]
        
        OpInfo['Ndn', i] = [[0, 0, 0, 0]
                           ,[0, 0, 0, 0]
                           ,[0, 0, 1, 0]
                           ,[0, 0, 0, 1]]
        
        OpInfo['N', i]  = [[0, 0, 0, 0]
                          ,[0, 1, 0, 0]
                          ,[0, 0, 1, 0]
                          ,[0, 0, 0, 2]]
        
        OpInfo['Sz', i] = [[0, 0, 0, 0]
                          ,[0, 1, 0, 0]
                          ,[0, 0,-1, 0]
                          ,[0, 0, 0, 0]]
    
        OpInfo['Sx', i] = [[0, 0, 0, 0]
                          ,[0, 0, 1, 0]
                          ,[0, 1, 0, 0]
                          ,[0, 0, 0, 0]]
        
        OpInfo['S-', i] = [[0, 0, 0, 0]
                          ,[0, 0, 0, 0]
                          ,[0, 1, 0, 0]
                          ,[0, 0, 0, 0]]
        
        OpInfo['S+', i] = [[0, 0, 0, 0]
                          ,[0, 0, 1, 0]
                          ,[0, 0, 0, 0]
                          ,[0, 0, 0, 0]]
        
    return BInfo, OpInfo