import torch
import torch.nn as nn
import torch.nn.functional as F

from network.convolution import FingerPrintBlock, SmilesConvBlock
from network.graph import GraphDenseNet


class AttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads,
    ):
        super(AttentionBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0
        
        self.q = nn.Linear(hidden_dim, hidden_dim)
        self.k = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, hidden_dim)
        
        self.ffn = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.5)
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim // num_heads]))
        
    def forward(self, query, key, value, mask):
        r"""
        Self Attention
        Both three Query, Key, Value come form the same source (For refining purpose)
        
        Parameters
        ----------
        query: torch.Tensor
            The query tensor.
        key: torch.Tensor
            The key tensor.
        value: torch.Tensor
            The value tensor.
        """
        Q = self.q(query)
        K = self.k(key)
        V = self.v(value)
        
        Q = Q.view(query.shape[0], self.n_heads, self.head_dim).unsqueeze(3)
        K = K.view(key.shape[0], self.n_heads, self.head_dim).unsqueeze(3)
        V = V.view(value.shape[0], self.n_heads, self.head_dim).unsqueeze(3)
        
        attention = self.dropout(F.softmax(
            torch.matmul(Q, K) / self.scale
        ))
        
        weight = torch.matmul(attention, V)
        weight = weight.permute(0, 2, 1, 3).contiguous()
        weight = weight.view(query.shape[0], self.num_heads * self.head_dim)
        weight = self.dropout(self.ffn(weight))
        
        return weight
    
# TODO: Implement the CrossAttentionBlock class
class CrossAttentionBlock(nn.Module):
    def __init__(
        self, 
        hidden_dim, 
        num_heads,
    ):
        super(CrossAttentionBlock, self).__init__()
        self.fp_encoder = FingerPrintBlock()
        self.smi_encoder = SmilesConvBlock()
        self.graph_encoder = GraphDenseNet()
        
        self.attention = AttentionBlock(hidden_dim, num_heads)
        
    def forward(self, fp, smi, graph):
        r"""
        Cross Attention
        The Query, Key, Value come from different sources (For fusion purpose)
        
        Parameters
        ----------
        fp_x: torch.Tensor
            A batch of 1D tensor for representing the information from fingerprint.
        graph_x: torch.Tensor
            A batch of 1D tensor for representing the infromation from graph.
        smi_x: torch.Tensor
            A batch of 1D tensor for represneting the information from smiles sequences.
        """
        fp = self.fp_encoder(fp)
        smi = self.smi_encoder(smi)
        graph = self.graph_encoder(graph)
        
        fp = self.attention(fp, fp, fp)
        smi = self.attention(smi, smi, smi)
        graph = self.attention(graph, graph, graph)
        
        return fp, smi, graph