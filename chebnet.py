'''
ChebNet: Efficient Graph Convolutional Neural Network with Edge-Aware Attention
https://arxiv.org/abs/1806.03536
https://arxiv.org/abs/1606.09375
'''

import numpy as np
import networkx as nx
import torch
from torch import nn
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
    else: # L = I-D^(-1/2) W D^(-1/2) (normalized)
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


'- Pytorch ChebConv layer -'

class ChebConv(nn.Module):
    def __init__(self, in_features, out_features, K: int):
        super(ChebConv, self).__init__()
        self.K = K # chebshev poly order
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(K, in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_param()
        
    def reset_param(self):
        nn.init.xavier_uniform_(self.weight.view(self.K* self.in_features, self.out_features))
        nn.init.zeros_(self.bias)
        
    def forward(self, x: torch.Tensor, L_scaled: torch.Tensor)->torch.Tensor:
        N = x.size(0)
        # compute chebyshev recurrence on the go
        Tx_prev2 = x
        Tx_prev1 = L_scaled @ x
        out = Tx_prev2 @ self.weight[0] # (N,F_in) @ (F_in,F_out) → (N,F_out)
        
        if self.K > 1:
            out += Tx_prev1 @ self.weight[1]
            
        for k in range(2, self.K):
            Tx_curr = 2*(L_scaled @ Tx_prev1) - Tx_prev2
            out += Tx_curr @ self.weight[k]
            Tx_prev2, Tx_prev1 = Tx_prev1, Tx_curr # recurrence shift window
            
        return out + self.bias
    

'- Graph pooling -'
'''
In image CNNs, pooling reduces spatial resolution (28x28 → 14x14) but on graphs, this is harder as nodes have no regular grid structure
The paper uses a graph coarsening algorithm (Graclus) that groups nodes into "super-nodes" by finding max-weight matchings iteratively
for simplicity here, we implement basic mean pooling over all nodes (global pooling) which gives a graph-level representation
'''

class GlobalPooling(nn.Module):
    # Shape: (N,F) → (1,F)  [or (B,N,F) → (B,F) for batched graphs]
    def forward(self, x: torch.Tensor)->torch.Tensor:
        return x.mean(dim=0, keepdim= True)
    

'- full chebnet classifier -'

class ChebNet(nn.Module):
    def __init__(self, in_features, hidden, num_classes, K = 3):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_features, hidden, K)
        self.conv2 = ChebConv(hidden, hidden, K)
        self.pool = GlobalPooling()
        self.classifier = nn.Linear(hidden, num_classes)
        
    def forward(self, x: torch.Tensor, L_scaled: torch.Tensor)->torch.Tensor:
        x = F.relu(self.conv1(x, L_scaled))
        x = F.relu(self.conv2(x, L_scaled))
        x = self.pool(x)
        x = self.classifier(x) # dim (1, num_classes)
        
        return F.log_softmax(x, dim=-1)
