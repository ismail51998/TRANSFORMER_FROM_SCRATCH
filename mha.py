import math
from typing import Optional, List
import torch
from torch import nn as nn
from labml import tracker
from labml_helpers.module import Module

#PREPARE FOR MULTI HEAD ATTENTION
#This module does a linear transformation and splits the vector into given number of heads for multi-head attention. 
# This is used to transform key, query, and value vectors.
class PrepareForMultiHeadAttention(Module):
    def __init__(self,d_model : int,heads: int, d_k: int,bias: bool):
        super.__init__()
        #Linear layer for linear transform
        self.linear = nn.Linear(d_model,heads*d_k,bias=bias)
        #number of heads
        self.heads=heads
        self.d_k=d_k
    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.heads, self.d_k)
        return x
#heads is the number of heads.
#d_model is the number of features in the query , key and value vectors.
class MultiHeadAttention(Module):
    def __init__(self, heads: int, d_model: int, dropout_prob: float = 0.1, bias: bool = True):
        super().__init__()
        #number of feature per head
        self.d_k = d_model//heads
        #number of heads
        self.heads=heads
#These transform the query , key and value vectors for multi-headed attention
        self.query = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.key = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=bias)
        self.value = PrepareForMultiHeadAttention(d_model, heads, self.d_k, bias=True)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(dropout_prob)
        self.scale = 1 / math.sqrt(self.d_k)
        self.attn = None
#Calculating the scores between queries and keys
    def get_scores(self,query: torch.Tensor,ley: torch.Tensor):
        #calculate QK^T
        return torch.einsum('ibhd,jbhd->ijbh',query,key)
#  mask has shape [seq_len_q, seq_len_k, batch_size] , 
# where first dimension is the query dimension. If the query dimension is equal to 1 it will be broadcasted.  
    def prepare_mask(self, mask: torch.Tensor, query_shape: List[int], key_shape: List[int]):
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]
        mask = mask.unsqueeze(-1)
        return mask
#query , key and value are the tensors that store collection of query, key and value vectors. They have shape [seq_len, batch_size, d_model] .
#mask has shape [seq_len, seq_len, batch_size] and mask[i, j, b] indicates whether for batch b , query at position i has access to key-value at position j .
    def forward(self, *,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        seq_len, batch_size, _ = query.shape
        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        scores = self.get_scores(query, key)
        scores *= self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.softmax(scores)
        tracker.debug('attn', attn)
        attn=self.dropout(attn)
        #the dropout can be done by adding mask filled with zeros that will
        #be multiplied by randomly by a head result
        #multiply by V
        x=torch.einsum("ijbh,jbhd->ibhd", attn, value)
        #EINSUM MAKE MULTIPLICATION BETWEEN DIM ARRAYS FOLLOWING THERE DIMS (i line j columns ..)
        self.attn = attn.detach()
        x = x.reshape(seq_len, batch_size, -1)
        return self.output(x)