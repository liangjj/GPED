import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from more_itertools import sort_together
import numpy as np

def setplt(SMALL_SIZE = 15, MEDIUM_SIZE = 20, BIGGER_SIZE = 25):
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
def lowestEigs(M, k = 10):
    E, V = eigsh(M.tocsc(), k = k , which = 'SA',maxiter=10000000)
    psi = []
    for i in range(len(V[0,:])):
        psi.append((np.transpose(np.matrix(V[:,i]))))
        
    E, psi = sort_together([E, psi])
    return E, psi

def fullEigs(M):
    E,V = eigh(M.toarray())   
    psi = []
    for i in range(len(V[0,:])):
        psi.append(np.conj(np.transpose(np.matrix(V[:,i]))))
    return E, psi

def overlap(psi1, psi2):
    return (np.transpose(np.conjugate(psi1))*psi2)[0,0]

def expVal(psi1, O, psi2):
    return (np.transpose(np.conjugate(psi1))*O.tocsc()*psi2)[0,0]

