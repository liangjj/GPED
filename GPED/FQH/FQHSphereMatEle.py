import numpy as np
from scipy.misc import factorial

# <j1,m1;j2,m2|j3,m3>
def cg_coeff(j1, j2, j3, m1, m2, m3):
    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) * factorial(j1 + m1) *
                factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C

def cg_key(j3,m1,m2,m3):
    return ','.join(str(e) for e in [j3,m1,m2,m3])

class cg_table:
    def __init__(self,Q):
        self.store = dict()
        # j1 = Q
        # j2 = Q
        # 0 <= j3 <= 2Q
        # -Q <= m1 <= Q
        # -Q <= m2 <= Q
        # -j3 <= m3 <= j3
        # key : cg_key(j3,m1,m2,m3)
        # value : <Q,m1;Q,m2|j3,m3>
        for j3 in range(int(2*Q+1)):
            for m3 in range(-j3,j3+1):
                for m1 in np.linspace(-Q,Q,int(2*Q+1)):
                    for m2 in np.linspace(-Q,Q,int(2*Q+1)):
                        key = cg_key(j3,m1,m2,m3)
                        self.store[key] = cg_coeff(Q,Q,j3,m1,m2,m3)
        
    def __getitem__(self,ks):#ks = [j3, m1, m2, m3]
        key = cg_key(ks[0],ks[1],ks[2],ks[3])
        return self.store[key]


# Haldane pseudopotential
class FQHSphereMatEle:
    def __init__(self, Nphi_, V_, cgtable):
        self.Nphi = Nphi_
        self.Q = Nphi_/2.0
        if len(V_) < self.Nphi:
            for c in range(len(V_), self.Nphi+2):
                V_.append(0)
        self.V = V_
        self.cgtable = cgtable
        
    def __getitem__(self, ms):
        if(ms[0] + ms[1] != ms[2] + ms[3]):
            return 0
        A = 0.0
        for L in range(self.Nphi+1):
            m_sum = int(ms[0]+ms[1])
            if(L >= abs(m_sum)):
                p = self.cgtable[L,ms[0],ms[1],m_sum]
                q = self.cgtable[L,ms[3],ms[2],m_sum]
                A += p*self.V[self.Nphi-L]*q;
        return A