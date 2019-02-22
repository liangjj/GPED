import numpy as np
# aspectRatio : Lx/Ly
class FQHTorusMatEle:
    def __init__(self, Ns, Vq, aspectRatio = 1, cutoff = 10, which = 'intra'):
        self.Ns = Ns
        self.Lx = np.sqrt(2*np.pi*Ns*aspectRatio)
        self.Ly = np.sqrt(2*np.pi*Ns/aspectRatio)
        self.Vq = Vq
        self.cutoff = cutoff
        self.which = which
        
        self.vkm = np.zeros((Ns, Ns), dtype = complex)
        for m in range(Ns):
            for n in range(Ns):
                self.vkm[n][m] = self._Vele(n, m)
        
    def _Vele(self, k, m):
        v = 0
        for q1 in range(-self.cutoff, self.cutoff+1):
            qx = 2*np.pi*q1/self.Lx
            for nm in range(-self.cutoff, self.cutoff+1):
                q2 = m + nm*self.Ns
                qy = 2*np.pi*q2/self.Ly
                q = np.sqrt(qx*qx+qy*qy)
                if q == 0 and self.which == 'intra':
                    continue
                v += self.Vq(qx,qy)*np.exp(-0.5*q*q)*np.cos(2*np.pi*q1*k/self.Ns)
        return v/self.Ns
    
    def __getitem__(self, ms):
        m1 = ms[0]
        m2 = ms[1]
        m3 = ms[2]
        m4 = ms[3]
        if( (m1+m2)%self.Ns != (m3+m4)%self.Ns ):
            return 0
        return self.vkm[(m1-m3)%self.Ns][(m1-m4)%self.Ns]
        