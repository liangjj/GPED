# GPED
A General Purpose Exact diagonalization library

GPED is an exact diagonalization(ED) code written in python for solving condensed matter second-quantized models 
with conserved quantum number. 

It is designed to be general and flexible. Moreover, it can construct the sparse matrix of an
operator by simply "typing" its second quatized form. 

## Example : 
### Tight binding model of 1D free spinless-Fermion.

In the following, I'll demonstrate how to construct the Hamiltonian matrix.

```
Nsite = 10 # number of sites

Ne = 1 # number of electrons

#Specify the Hilbert space and operators

BasisInfo, OperatorInfo = SpinlessFermion(Nsite)

#Generate the Basis

AllBasis = BasisSet(BasisInfo, Ne)

#Write Down the Hamiltonian

h = OperatorMat(OpInfo)

for i in range(N-1):
    h += [1, 'Cdag', i, 'C', i+1]
    h += [1, 'Cdag', i+1, 'C', i, ]
    
# convert the Hamintonian to sparse matrix format
H = getMat(h, AllBasis)
```
