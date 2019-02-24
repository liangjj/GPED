import numpy as np
from GPED.BasisInfo import BasisInfo
from GPED.OperatorInfo import OperatorInfo
from GPED.OperatorMat import OperatorMat
from GPED.utils import setplt, expVal
from GPED.getMat import getMat

import matplotlib.pyplot as plt

# Q : monopole strength
def FQHSphere(Q):
    
    if((2*Q)%1 > 1E-8):
        raise RunTimeError
        
    N = int(2*Q + 1)
    Nphi = int(2*Q)
    
    BInfo = BasisInfo(['0','1'], N)
    for i in range(N):
        BInfo.put(i, '0', [0, 0])
        BInfo.put(i, '1', [1, i-Nphi/2.0])
        
    OpInfo = OperatorInfo(BInfo)
    
    for i in range(N):
        OpInfo['n',i] = [[0, 0]
                        ,[0, 1]]
    
        OpInfo['C',i] = [[0, 1]
                        ,[0, 0]]
    
        OpInfo['Cdag',i] = [[0, 0]
                           ,[1, 0]]
        
    
    return BInfo, OpInfo


def FQHSphere_bilayer(Q, conserveSz = True):
    
    if((2*Q)%1 > 1E-8):
        raise RunTimeError
        
    N = int(2*Q + 1)
    Nphi = int(2*Q)
    
    # state of a site
    # 0 : vac
    # u : 1 electron in upper layer
    # d : 1 electron in lower layer
    # 2 : 2 electrons in both layers
    statelist = ['0','u', 'd', '2']
    
    BInfo = BasisInfo(statelist, N)
    if(conserveSz):
        for i in range(N):
            BInfo.put(i, '0', [0,              0, 0])
            BInfo.put(i, 'u', [1,     i-Nphi/2.0, 1])
            BInfo.put(i, 'd', [1,     i-Nphi/2.0,-1])
            BInfo.put(i, '2', [2, 2*(i-Nphi/2.0), 0])
    else:
        for i in range(N):
            BInfo.put(i, '0', [0,              0])
            BInfo.put(i, 'u', [1,     i-Nphi/2.0])
            BInfo.put(i, 'd', [1,     i-Nphi/2.0])
            BInfo.put(i, '2', [2, 2*(i-Nphi/2.0)])
            
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

def Apls(l,m):
    if(abs(m) > l):
        return 0
    return np.sqrt(l*(l+1)-m*(m+1))

def Amns(l,m):
    if(abs(m) > l):
        return 0
    return np.sqrt(l*(l+1)-m*(m-1))

def L2_eig2l(L2): # convert the eigenvalue of L2 to l
    l = (-1 + np.sqrt(1+4*L2))/2
    if(abs(l.imag > 1E-8)):
        raise RuntimeError
    l = round(l,1)
    return l

def Ltotal(Q, OpInfo, BSet, psis, Lz):
    # total angular momentum of energy eigenstate psis
    # L^2 = L+L- + L_z^2 - L_z
    # Since Lz is proportional to identity matrix by construction
    # ,we only build L+L- matrix.

    Nphi = int(2*Q)
    
    LpLm_Op = OperatorMat(OpInfo)

    # L+L- diagonal part
    for l in range(Nphi+1):
        m = l-Q
        LpLm_Op += [Apls(Q, m-1)*Amns(Q, m), 'n', l];

    # L+L- off-diagonal part
    for l1 in range(Nphi+1):
        for l2 in range(Nphi+1):
            #if(l1 != l2 and l1+1 != l2-1):
            m1 = l1 - Q
            m2 = l2 - Q
            LpLm_Op += [Apls(Q, m1)*Amns(Q, m2), 'Cdag', l1+1, 'Cdag', l2-1, 'C', l2, 'C', l1]

    LpLm = getMat(LpLm_Op, BSet)
    
    L2_list = []
    
    for psi in psis:
        e = expVal(psi, LpLm, psi) - Lz**2 - Lz
        L2_list.append(L2_eig2l(e))
    return L2_list

def Ltotal_bilayer(Q, OpInfo, BSet, psis, Lz):
    # total angular momentum of energy eigenstate psis
    # L^2 = L+L- + L_z^2 - L_z
    # Since Lz is proportional to identity matrix by construction
    # ,we only build L+L- matrix.

    LpLm_Op = OperatorMat(OpInfo)
    Nphi = int(2*Q)
    
    # L+L- diagonal part
    for l in range(Nphi+1):
        m = l-Q
        LpLm_Op += [Apls(Q, m-1)*Amns(Q, m), 'N', l];

    # L+L- off-diagonal part
    for l1 in range(Nphi+1):
        for l2 in range(Nphi+1):
            m1 = l1 - Q
            m2 = l2 - Q
            LpLm_Op += [Apls(Q, m1)*Amns(Q, m2), 'Cdagup', l1+1, 'Cdagup', l2-1, 'Cup', l2, 'Cup', l1]
            LpLm_Op += [Apls(Q, m1)*Amns(Q, m2), 'Cdagdn', l1+1, 'Cdagdn', l2-1, 'Cdn', l2, 'Cdn', l1]
            LpLm_Op += [Apls(Q, m1)*Amns(Q, m2), 'Cdagup', l1+1, 'Cdagdn', l2-1, 'Cdn', l2, 'Cup', l1]
            LpLm_Op += [Apls(Q, m1)*Amns(Q, m2), 'Cdagdn', l1+1, 'Cdagup', l2-1, 'Cup', l2, 'Cdn', l1]

    LpLm = getMat(LpLm_Op, BSet)
    
    L2_list = []
    for psi in psis:
        e = expVal(psi, LpLm, psi) - Lz**2 - Lz
        L2_list.append(L2_eig2l(e))
    return L2_list


def plot_ES_conserveSz(q,s,ES):
    K = []
    xi = []
    for es in ES:
        if es[0][0] == q and es[0][2] == s:
            K.append(es[0][1])
            xi.append(es[1])
    
    plt.scatter(K, xi, alpha = 0.5)
    
    
def plot_ES(q,ES):
    setplt()
    K = []
    xi = []
    for es in ES:
        if es[0][0] == q:
            K.append(es[0][1])
            xi.append(es[1])
    
    plt.scatter(K, xi, alpha = 0.5)
