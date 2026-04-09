'''
Graph Attention Networks (GAT) -
https://arxiv.org/abs/1710.10903
'''

import torch
from torch import nn
import torch.nn.functional as F


'- GAT layer -'
class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads = 1, concat = True, dropout = 0.6):
        super().__init__()
        self.in_features = in_features
        self.num_heads = num_heads
        self.out_features = out_features
        self.concat = concat
        self.dropout = dropout
        # linear projection
        self.W = nn.Linear(in_features, num_heads* out_features, bias= False)
        # attention vector a
        self.a_src = nn.Parameter(torch.empty(num_heads, out_features))
        self.a_dst = nn.Parameter(torch.empty(num_heads, out_features))
        
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout_layer = nn.Dropout(p=dropout)
        
        # optional output proj when avg heads (only if concat not true)
        self.bias = nn.Parameter(torch.zeros(num_heads*out_features if concat else out_features))
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        
    def forward(self, h: torch.Tensor, adj: torch.Tensor)->torch.Tensor:
        N = h.size(0) #(N, in_features)
        Wh = self.W(h).view(N, self.num_heads, self.out_features) #(N, num_nodes, out_features)
        
        # for each edge(i,j) and each head k, e(i,j) = LeakyRelu(a_src[k].Wh[i,k] + a_dst[k].Wh[j,k])
        e_src = (Wh* self.a_src).sum(dim=-1) #(N, num_heads)
        e_dst = (Wh* self.a_dst).sum(dim=-1) #(N, num_heads)
        
        # broadcast e_src as rows and e_dst as column instead of iterating over all edges
        e = self.leaky_relu(e_src[:,None,:] + e_dst[None,:,:]) # e_src(N, 1, num_heads) and e_dst(1, N, num_heads) -> result (N, N, num_heads)
        
        # masking where adj[i,j] is 1 (neighbors)
        mask = (adj == 0).unsqueeze(-1) 
        e_masked = e.masked_fill(mask, float('-inf')) # all non neighbors to 0
        
        alpha = F.softmax(e_masked, dim=1)
        alpha = self.dropout_layer(alpha)
        
        # weighted agg
        h_prime = (alpha[:,:,:,None]* Wh[None]).sum(dim=1) # (N, num_heads, out_features)
        
        if self.concat:
            h_prime = h_prime.reshape(N, self.num_heads* self.out_features)
        else:
            h_prime = h_prime.mean(dim=1)
        
        return h_prime + self.bias
    

'- GAT Model -'

class GAT(nn.Module):
    def __init__(self, in_features, hidden_dim = 8, num_classes = 7, num_heads = 8, dropout = 0.6):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATLayer(in_features=in_features,
                             out_features=hidden_dim,
                             num_heads=num_heads,
                             concat=True,
                             dropout=dropout)
        
        self.gat2 = GATLayer(in_features=num_heads* hidden_dim,
                             out_features=num_classes,
                             num_heads=1,
                             concat=False,
                             dropout=dropout)
        
    def forward(self, x: torch.Tensor, adj: torch.Tensor)->torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.gat1(x, adj))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, adj)
        
        return F.log_softmax(x, dim=1)
    
    
' some training utilities or improvs on graph (not related to GAT but might improve performance) '
        
def self_loops(adj: torch.Tensor)->torch.Tensor:
    N = adj.size(0)
    return adj + torch.eye(N, device=adj.device) # adding self loops

def normalize_adj(adj: torch.Tensor)->torch.Tensor:
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.0
    D = torch.diag(deg_inv_sqrt)
    
    return D @ adj @ D
