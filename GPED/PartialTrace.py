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
        self.keep_region = keep_region
        
        ## reduced BSet : p1 -> index
        BSet_reduced_store = dict()
        
        ## PTInfo[p2] = (index, index_reduced)
        self.PTInfo = dict()
        
        for b in BSet.store.keys():
            p1, p2 = split_basis(BSet.BasisInfo, b, keep_region)
            
            if p2 not in self.PTInfo:
                self.PTInfo[p2] = []
                
            self.PTInfo[p2].append((BSet[b], BSet_reduced_store[p1]))
            
            if p1 not in BSet_reduced:
                BSet_reduced_store[p1] = len(BSet_reduced_store)
                
        self.BSet_reduced = BasisSet(BSet.BasisInfo, BSet_reduced_store)
        
    def __getitem__(self, rho):
        
        if rho.shape[1] == 1:
            rho = rho*np.transpose(np.conjugate(rho))
            
        rho_reduced = np.matrix(np.zeros((len(self.BSet_reduced), len(self.BSet_reduced)), dtype=complex))
        
        for p2 in self.PTInfo.keys():
            for (row_index, row_index_reduced) in self.PTInfo[p2]:
                for (col_index, col_index_reduced) in self.PTInfo[p2]:
                    rho_reduced[row_index_reduced, col_index_reduced] += rho[row_index, col_index]
                
        return rho_reduced

