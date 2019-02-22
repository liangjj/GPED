import numpy as np
from Basis import QN, BasisInfo

def QNofLeftBasis(Leftbket, BasisInfo):
    offset = BasisInfo.numBitPerSite
    
    qn = BasisInfo[0][BasisInfo.stateSet[Leftbket%(1<<offset)]]
    for pos in range(1,BasisInfo.Nsite):
        Leftbket = Leftbket >> offset
        qn = qn + BasisInfo[pos][BasisInfo.stateSet[Leftbket%(1<<offset)]]
    
    return qn

def QNstrtolist(s):
    L = []
    s = s[1:len(s)-1]
    pos = s.find(',')
    while pos > 0:
        L.append(float(s[0:pos]))
        s = s[pos+1:len(s)]
        pos = s.find(',')
    L.append(float(s))
    return QN(L)


def EntanglementSpectrum(psi, BSet, BInfo, N_L):
    N_R = BInfo.Nsite - N_L
    D = len(BSet)

    getRbasis = lambda ket : (ket >> N_R*BInfo.numBitPerSite) << N_R*BInfo.numBitPerSite
    getLbasis = lambda ket : ket%(1<<N_R*BInfo.numBitPerSite)

    # sort the psi and basis as 
    # QN ---dict--> Rbasis ---dict--> [Lbasis, coefficient]
    cont = dict()
    #index = 0
    for b in BSet.getBasis():
        index = BSet[b]
        Lb = getLbasis(b)
        Rb = getRbasis(b)
        qnL = QNofLeftBasis(Lb, BInfo)
    
        if(str(qnL) in cont):
            if Rb in cont[str(qnL)]:
                cont[str(qnL)][Rb].append([Lb, psi[index]])
            else :
                cont[str(qnL)][Rb] = [[Lb, psi[index]]]
        else:
            cont[str(qnL)] = dict()
            cont[str(qnL)][Rb] = [[Lb, psi[index]]]

        #index = index + 1
    
    # diagonalize the density matrix for each block of quantum number
    ES = []
    SvN = 0
    for qn in cont.keys():

        #density matrix of a given quanutm number block
        rho_ = np.matrix([0], dtype = float)

        #make a table for Lbasis -> matrix index
        LbasistoMatIndex = dict()

        for Rb in cont[qn].keys():
            # make a ket vector for a given Rbasis
            psi_ = np.matrix([0], dtype = float)
            dim = 1
            LbCoeffSet = cont[qn][Rb]
            for LbCoeff in LbCoeffSet:
                Lb = LbCoeff[0]
                Coeff = LbCoeff[1]
                if(not Lb in LbasistoMatIndex):
                    LbasistoMatIndex[Lb] = len(LbasistoMatIndex)
                if LbasistoMatIndex[Lb] >= dim:
                    psi_tem = psi_.copy()
                    psi_ = np.matrix(np.zeros((LbasistoMatIndex[Lb]+1,1)), dtype = float)
                    for i in range(len(psi_tem)):
                        psi_[i] = psi_tem[i] 
                    dim = len(psi_)

                psi_[LbasistoMatIndex[Lb],0] = psi_[LbasistoMatIndex[Lb],0] + np.real(Coeff)

            rhotem = psi_*np.transpose(psi_)
            if(len(rho_) < len(rhotem)):
                np.resize(rho_,(len(rhotem),len(rhotem)))
            rho_ = rho_ + rhotem
        eig = np.linalg.eigh(rho_)
        
        for e in eig[0]:
            if(abs(e)>1E-10):
                s = -e*np.log(e)
                SvN += s
            
        for e in eig[0]:
            if(abs(e)>1E-10):
                ES.append([QNstrtolist(qn), -np.log(e)])
    
    return SvN, ES
        