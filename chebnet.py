'''
ChebNet: Efficient Graph Convolutional Neural Network with Edge-Aware Attention
https://arxiv.org/abs/1806.03536
https://arxiv.org/abs/1606.09375
'''

import numpy as np
import networkx as nx
import scipy.sparse as sp
import torch
from torch import nn, optim
import torch.nn.functional as F
import matplotlib.pyplot as plt


'- Graph basics performed -'

def graph_laplacian(adj_m: np.ndarray, normalized = True)->np.ndarray:
    # adj_matrix dim (N,N) and returns the same dim laplacian matrix
    N = adj_m.shape[0]
    deg = adj_m.sum(axis=1)
    if not normalized: # L = D-W (unnormalized)
        D = np.diag(deg)
        L = D - adj_m
    else: # L = I-Di^(-1/2).W.Dj^(-1/2) (normalized)
        d_inv_sqrt = np.where(deg>0, 1.0/np.sqrt(deg), 0.0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        L = np.eye(N) - D_inv_sqrt @ adj_m @ D_inv_sqrt
        
    return L

def laplacian_spectrum(L: np.ndarray):
    # decompose the laplacian L = U Λ U^T where Λ is eigen value and U is eigen vectoor
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    return eigenvalues, eigenvectors # dim (N) and (N,N)


'- Graph fourier transform and spectral conv -'
'''
graph fourier transform: x_hat = U^T x (project signal onto eigenvector basis)
inverse GFT: x = U x_hat (reconstruct signal from spectral coeff)
spectral convolution: y = g_θ * x  =  U (g_θ(Λ) ⊙ (U^T x))
where g_θ(Λ) is a learnable filter in the spectral domain, ⊙ = element-wise multiplication (applying filter to each frequency)
'''

def naive_spectral_conv(x: np.ndarray, U: np.ndarray, filter_weights: np.ndarray)-> np.ndarray:
    # y = U* diag(g_theta)* U^T * x
    x_hat = U.T @ x
    x_hat_filtered = filter_weights* x_hat # apply filter for filter domain
    y = U @ x_hat_filtered # IFT back to spatial domain
    
    return y


'- Chebyshev polynomial approximation -'
'''
solving the problems with naive spectral conv - too much computing O(N^3) and storing O(N^2), filters not localized in space and dont generalize
so convolution becomes: y = g_θ(L) x = Σ_{k=0}^{K-1}  θ_k * T_k(L_scaled) * x
where T_k are Chebyshev polynomials of the first kind, and L_scaled rescales
'''

def chebyshev_polynomial(L: np.ndarray, K: int)->list:
    N = L.shape[0]
    lambda_max = 2.0
    L_scaled = (2.0/ lambda_max)* L - np.eye(N) # should be in [-1,1] for chebyshev poly
    
    T = []
    T.append(np.eye(N))
    if K>1:
        T.append(L_scaled.copy())
    
    for k in range(2, K): #recurrence
        Tk = 2* L_scaled @ T[k-1] - T[k-2]
        T.append(Tk)
        
    return T # T[k] is (N,N) matrix for the k-th Chebyshev polynomial

def cheb_conv_numpy(x: np.ndarray, cheb_basis: list, theta: np.ndarray)->np.ndarray:
    # apply a ChebConv filter to signal x using pre-computed Chebyshev basis
    K = len(cheb_basis)
    F_in = x.shape[1]
    F_out = theta.shape[2]
    N =x.shape[0]
    
    y = np.zeros((N, F_out))
    for k in range(K):
        Tx = cheb_basis[k] @ x
        y += Tx @ theta[k] # (N,F_in) @ (F_in,F_out) → (N,F_out)
        
    return y

