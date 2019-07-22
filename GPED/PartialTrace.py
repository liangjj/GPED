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
        self.BSet_reduced = dict()
        #
        self.PTInfo = dict()
        
        # PTInfo[p2] = (index, index_reduced)
        for b in BSet.store.keys():
            p1, p2 = split_basis(self.BSet.BasisInfo, b, keep_region)
            
            if p2 not in self.PTInfo:
                self.PTInfo[p2] = []
            if p1 not in self.BSet_reduced:
                self.BSet_reduced[p1] = len(self.BSet_reduced)
                
            self.PTInfo[p2].append((BSet[b], self.BSet_reduced[p1]))
            
            
    def __getitem__(self, rho):
        
        if rho.shape[1] == 1:
            rho = rho*np.transpose(np.conjugate(rho))
            
        rho_reduced = np.matrix(np.zeros((len(self.BSet_reduced), len(self.BSet_reduced)), dtype=complex))
        
        for p2 in self.PTInfo.keys():
            for (row_index, row_index_reduced) in self.PTInfo[p2]:
                for (col_index, col_index_reduced) in self.PTInfo[p2]:
                    rho_reduced[row_index_reduced, col_index_reduced] += rho[row_index, col_index]
                
        return rho_reduced

