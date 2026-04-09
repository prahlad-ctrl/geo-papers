'''
Graph Convolutional Networks (GCN) -
https://arxiv.org/abs/1609.02907
'''

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


'- GCN normalization -'

def gcn_norm_adj(adj_m: np.ndarray)->np.ndarray:
    N = adj_m.shape[0]
    A_hat = adj_m + np.eye(N) # the key part of gcn is self looping before norm (A+I)
    deg = A_hat.sum(axis=1)
    d_inv_sqrt = np.where(deg>0, 1.0/np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt # D^(-1/2) A_hat D^(-1/2)
    
    return A_norm

# this is different from graph laplacian as we self loop for gcn before norm so that we include own features + neighbors


'- GCN Conv layer -'

class GCNConv(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weights = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.reset_param()
        
    def reset_param(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)
        
    def forward(self, x:torch.Tensor, A_norm: torch.Tensor)-> torch.Tensor:
        b_x = x @ self.weights
        H = A_norm @ b_x
        # H = A_hat @ x @ W ---- [(N,N) @ (N,F_in) @ (F_in,F_out) → (N,F_out)]
        
        return H + self.bias
    
    
'- Global pooling -'

class GlobalPooling(nn.Module):
    def forward(self, x:torch.Tensor)-> torch.Tensor:
        return x.mean(dim=0, keepdim= True) #(N,F) → (1,F)
    
    
'- full GCN classifier -'

class GCN(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.pool = GlobalPooling()
        self.classifier = nn.Linear(hidden, num_classes)
        
    def forward(self, x: torch.Tensor, A_norm: torch.Tensor)->torch.Tensor:
        x = F.relu(self.conv1(x, A_norm))
        x = F.relu(self.conv2(x, A_norm))
        x = self.pool(x)
        x = self.classifier(x) # dim (1, num_classes)
        
        return F.log_softmax(x, dim=-1)
    