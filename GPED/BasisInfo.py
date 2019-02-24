import numpy as np

'''
class BasisInfo

constructor(stateSet, Nsite)

------information about hilber space------- 

__getitem__(position)

put(position, state, qn)

getState()

sumQN(ket)

__len__

stateSet

------convert kets between string and binary------

getBinatPosition(position, ket)

toBinRepLocal(position, state)

toBinRep(ket)

toStrRep(ketbin)

--------------------------------------------------

Nsite

numBitPerSite

stateSet
'''

'''
class QN

__add__(other)

__sub__(other)
'''


'''    
class locHbt 

constructor()

__setitem__(state, qn)

__getitem__(s) s -> qn

dim()
'''

class QN(list):
    
    def append(self,n):
        raise NotImplemented
    
    def pop(self):
        raise NotImplemented
        
    def __add__(self, other):
        if(len(self) != len(other)):
             raise RuntimeError
        r = []
        for i in range(len(self)):
            r.append(self[i] + other[i])
                     
        return QN(r)
    
    def __sub__(self, other):
        if(len(self) != len(other)):
            raise RuntimeError
        r = []
        for i in range(len(self)):
            r.append(self[i] - other[i])
                     
        return QN(r)
    
    def __eq__(self, other):
        if(len(self) != len(other)):
            return False
        for i,j in zip(self, other):
            if(i != j):
                return False
        return True
    
    def __ne__(self,other):
        return not(self == other)
    
class _locHbt:
    
    def __init__(self):
        self.store = dict()
            
    def __getitem__(self, s):
        return self.store[s]
    
    def __setitem__(self,state, qn):
        if(len(state) > 1):
            raise RuntimeError  
        self.store[state] = QN(qn)
    
    def dim(self):
        return len(self.store)
    
    def __str__(self):
        msg = '/-----------------------------------\\\n'
        for state in self.store.keys():
            msg = msg +'  '+ 'state : |' + state +'>'+ '\tQN : ' + str(self.store[state]) + '\n'
        msg = msg + '\\-----------------------------------/'
        return msg
    
    

class _Rep:
    def __init__(self, stateSet, Nsite):
        self.stateSet = stateSet
        self.numBitPerSite = int(np.ceil(np.log(len(stateSet))/np.log(2)))
        self.Nsite = Nsite
        
    def getBinatPosition(self, position, ket):
        return (((ket >> self.numBitPerSite*position)%(1 << self.numBitPerSite))
                                                            << self.numBitPerSite*position)
    
    def toBinRepLocal(self, position, state):
        return self.stateSet.index(state) << self.numBitPerSite * position
    
    
    def toBinRep(self, ket):
        b = 0
        for i in range(len(ket)):
            b = b + self.toBinRepLocal(i, ket[i])
        return b
    
    def toStrRep(self, ketbin):
        state = ''
        while ketbin > 0:
            index = ketbin%(2**(self.numBitPerSite))
            state = state + self.stateSet[index]
            ketbin = ketbin >> self.numBitPerSite
        
        while len(state) < self.Nsite:
            state = state + '0'     
        return state
    
class _Hbt:
    def __init__(self, stateSet):
        self.store = dict()
        self.stateSet = stateSet
        
    def __getitem__(self, position):
        return self.store[position]
    
    def put(self, position, state, qn):
        if(type(qn) == int):
            qn = [qn]
        if position in self.store:
            self.store[position][state] = QN(qn)
        else:
            self.store[position] = _locHbt()
            self.store[position][state] = QN(qn)
            
    def getState(self):
        return self.stateSet
    
    def __len__(self):
        return len(self.store)
    
    def __str__(self):
        msg = "Hilber Space\n\n"
        for i in range(len(self.store)):
            msg = msg + 'Site no.%d\n'%i
            msg = msg + str(self.store[i]) + '\n\n'
        return msg
    
    
class BasisInfo(_Rep, _Hbt):
    def __init__(self, stateSet, Nsite):
        _Rep.__init__(self, stateSet, Nsite)
        _Hbt.__init__(self, stateSet)
       
    
    def getQN(self, ket):
        sket = self.toStrRep(ket)
        qnlist = []
        for i in range(self.Nsite):
            qnlist.append(self[i][sket[i]])
    
        sumqn = QN([0]*len(qnlist[0]))
        for qn in qnlist:
            sumqn = sumqn + qn
        return sumqn