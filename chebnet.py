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

