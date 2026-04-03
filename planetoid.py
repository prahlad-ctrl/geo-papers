'''
Planetoid: Inductive Representation Learning on Large Graphs -
https://arxiv.org/abs/1706.02216
'''


import numpy as np
import torch
from torch import nn

'''
Planetoid - Transductive -
the entire graph structure and all node features available for training
'''

class PlanetoidT(nn.Module):
    def __init__(self, n_nodes, n_features, n_classes, emb_dim = 64, hidden = 128):
        super().__init__()
        self.emb_dim = emb_dim
        self.node_emb = nn.Embedding(n_nodes, emb_dim) # every node has its own emb vector
        nn.init.xavier_uniform_(self.node_emb.weight)
        
        self.clf = nn.Sequential(nn.Linear(n_features + emb_dim, hidden),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden, n_classes))
    
    def forward_supervised(self, x, node_idx):
        e = self.node_emb(node_idx) # dim(batch_size, emb_dim)
        h = torch.cat([x, e],dim=-1) # dim(batch_size, n_features + emb_dim)
        
        return self.clf(h)
    
    def forward_context(self, anchor_idx, context_idx):
        ea = self.node_emb(anchor_idx) # same dim(batch_size, emb_dim)
        ec = self.node_emb(context_idx)
        score = (ea*ec).sum(dim=-1) # dot product , dim(batch_size)

        return torch.sigmoid(score)
        

'''
Planetoid - Inductive -
designed to generalize on unseen nodes or entirely new graphs
'''

class PlanetoidI(nn.Module):
    def __init__(self, n_features, n_classes, emb_dim=64, hidden= 128):
        super().__init__()
        self.emb_dim = emb_dim
        # features -> embedding
        self.encoder = nn.Sequential(nn.Linear(n_features, hidden),
                                     nn.ReLU(),
                                     nn.Linear(hidden, emb_dim))
        # embedding -> class
        self.cls = nn.Sequential(nn.Linear(emb_dim, hidden),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(hidden, n_classes))
    
    def embed(self, x):
        return self.encoder(x)
    
    def forward_supervised(self, x):
        e = self.embed(x)
        return self.clf(e)
    
    def forward_context(self, x_anchor, x_context):
        ea = self.embed(x_anchor)
        ec = self.embed(x_context)
        score = (ea*ec).sum(dim=-1)
        
        return torch.sigmoid(score)
