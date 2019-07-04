from GPED.BasisSet import BasisSet
import scipy
import numpy as np


## helper function to split the basis
def split_basis(BInfo, ket, keep_region):
    p1 = 0
    p2 = 0
    curr = ket

    offset = BInfo.numBitPerSite

    for i in range(BInfo.Nsite):
        if i in keep_region:
            p1 += (curr%(1<<offset)) << i*offset
        else:
            p2 += (curr%(1<<offset)) << i*offset
        curr = curr >> offset
    return p1, p2

class PartialTrace:
    def __init__(self, BSet, keep_region):
        ## BSet_split : bset --dict--> p1,p2
        ## reduced BSet : p1 -> index
        self.BSet = BSet
        self.keep_region = keep_region
        self.offset = BSet.BasisInfo.numBitPerSite
        self.BSet_split = dict()
        self.BSet_reduced = dict()
        for b in BSet.store.keys():
            self.BSet_split[b] = split_basis(self.BSet.BasisInfo, b, keep_region)
            if self.BSet_split[b][0] not in self.BSet_reduced:
                self.BSet_reduced[self.BSet_split[b][0]] = len(self.BSet_reduced)

    def __getitem__(self, psi):
        rho = scipy.sparse.lil_matrix((len(self.BSet_reduced), len(self.BSet_reduced)), dtype=complex)

        for row in self.BSet.store.keys():
            for col in self.BSet.store.keys():
                i = self.BSet[row]
                j = self.BSet[col]
                row_p1, row_p2 = self.BSet_split[row]
                col_p1, col_p2 = self.BSet_split[col]

                if row_p2 == col_p2:
                    rho[self.BSet_reduced[row_p1], self.BSet_reduced[col_p1]] += np.conj(psi[i][0,0])*psi[j][0,0]

        return rho

