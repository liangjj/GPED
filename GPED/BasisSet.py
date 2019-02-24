from GPED.BasisInfo import QN, BasisInfo
import numpy as np
from scipy.special import comb

##################################################################################
# check_succeed methods can be overided for different system
# default_check_succeed return true when all the qunatum number are equal to qn_goal

def default_check_succeed(qn_curr, qn_goal , BasisInfo = []):
    return qn_curr == qn_goal
    

'''
class BasisSet

Constructor(BasisInfo, QN) 

getBasis()                

__getitem__(key)           

'''

class BasisSet:
    def __init__(self, basisinfo, qn = [0], check_succeed = default_check_succeed, store = None):
        
        self.BasisInfo = basisinfo
        
        if store is None:
            if(type(qn) == int):
                qn = [qn]
            
        
            self.qn_goal = QN(qn)
            self.check_succeed = check_succeed
        
            #################################################################################
            ## All the basis with the same number of particles can be thought of as a graph.
            ## Each node of the graph are the basis itself. 
            ## Two nodes |a> and |b> are connected if and only if there exist an operator
            ## A = c^-_i*c^+_{i+1} and 0 <= i < N-1 such that |b> = A|a> 
            ## where c^-_i is annihilation operator at site i and c^+_{i+1} is creation 
            ## operator at site i+1.
            ## 
            ## We use DFS to traverse the whole graph to find the state that satisfies 
            ## self.check_succeed(basis) == True
            self.store = self.dfs()
        else:
            self.store = store
            
    def getBasis(self):
        return self.store.keys()
    
    def __getitem__(self, key):
        return self.store[key]
    
    def __len__(self):
        return len(self.store)
    
    def __str__(self):
        
        #############################################
        ## Sort the Basis accroding to its label
        orderStore = dict()
        for b in self.store.keys():
            orderStore[self.store[b]] = b
        s = 'label\tStr\tBin\n'
        for b in orderStore.keys():
            s = s + str(b)+'\t' + self.BasisInfo.toStrRep(orderStore[b]) + '\t' + str(orderStore[b]) + '\n'
        return s
    
    def __contains__(self, k):
        return k in self.store
    
    # initialize the roots
    def findroots(self):
        num = self.qn_goal[0]
        # make a dictionary : numofParticle -> stateList
        stateSet =self.BasisInfo.getState()
        maxnum = 0
        M = dict()
        for s in stateSet:
            if(maxnum <self.BasisInfo[0][s][0]):
                maxnum =self.BasisInfo[0][s][0]
            numP =self.BasisInfo[0][s][0]
            if numP in M:
                M[numP].append(s)
            else:
                M[numP] = [s]

        # recursive fucntion to find the roots
        def findroots_util(pos, res, root, rootList):
            if(res == 0):
                rootList.append(root)
                return
            if(res >= maxnum):
                for state in M[maxnum]:
                    findroots_util(pos+1, res-maxnum, root +self.BasisInfo.toBinRepLocal(pos, state), rootList)
            else :
                for state in M[res]:
                    findroots_util(pos+1, 0, root +self.BasisInfo.toBinRepLocal(pos, state), rootList)

        rootList = []
        findroots_util(0, num, 0, rootList)
        return rootList

    def dfs(self):
        offset = self.BasisInfo.numBitPerSite
        ############################################################################
        # moveing a particle can be implemented by adding 
        # and substracting number in binary repersentation.
        # So we make a table to define the rule for increasing and decreasing 
        # number of particle by 1. 
        # i.e. to decereasing the number of particle at site i by 1:
        # ket = ket + (decreaseNumBy1[ket] << i*numBitPerSite)
        decreaseNumBy1 = dict()
        increaseNumBy1 = dict()
        for fromstate in self.BasisInfo.stateSet:
            for tostate in self.BasisInfo.stateSet:
                bfrom = self.BasisInfo.toBinRepLocal(0,fromstate)
                bto = self.BasisInfo.toBinRepLocal(0,tostate)
                if(self.BasisInfo[0][fromstate][0] - self.BasisInfo[0][tostate][0] == 1):
                    if(bfrom in decreaseNumBy1):
                        decreaseNumBy1[bfrom].append(bto - bfrom)
                    else:
                        decreaseNumBy1[bfrom] = [bto - bfrom]
                elif(self.BasisInfo[0][fromstate][0] - self.BasisInfo[0][tostate][0] == -1):
                    if(bfrom in increaseNumBy1):
                        increaseNumBy1[bfrom].append(bto - bfrom)
                    else:
                        increaseNumBy1[bfrom] = [bto - bfrom]
                        
        ######################################################################
        def neighbor(ket):
            neigList = []
            curr = ket
            nex = ket >> offset
            for i in range(self.BasisInfo.Nsite-1):
                BitAt_i = curr % (1 << offset)
                BitAt_ipls1 = nex % (1 << offset) 
        
                if((BitAt_i in decreaseNumBy1) and (BitAt_ipls1 in increaseNumBy1)):
                    Ldec = decreaseNumBy1[BitAt_i]
                    Linc = increaseNumBy1[BitAt_ipls1]
                    for dec in Ldec:
                        for inc in Linc:
                            decAtpos_i = dec << i*offset
                            incAtpos_i = inc << (i+1)*offset
                            neigList.append(ket + decAtpos_i + incAtpos_i)
                curr = curr >> offset
                nex = nex >> offset
            return neigList
        
        ######################################################################
        #
        #
        def dfs_util(root, visited, allBasis):
            if root in visited:
                return
            else :
                
                visited.add(root)
                if(self.check_succeed(self.BasisInfo.getQN(root), self.qn_goal, self.BasisInfo)):
                    allBasis[root] = len(allBasis)
            
            for neig in neighbor(root):
                dfs_util(neig, visited, allBasis)
                
        ######################################################################
        
        rootList = self.findroots()
        visited = set()
        allBasis = dict()
        for root in rootList:
            dfs_util(root, visited, allBasis)
        
        return allBasis