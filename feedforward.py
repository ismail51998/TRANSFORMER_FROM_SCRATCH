import torch
from torch import nn as nn
from labml_helpers.module import Module
'''

'''
class FeedForward(Module):
#d_model is the number of features in a token embedding
#d_ff is the number of features in the hidden layer of the FFN
#dropout is dropout probability for the hidden layer
#is_gated specifies whether the hidden layer is gated
#bias1 specified whether the first fully connected layer should have a learnable bias
#bias2 specified whether the second fully connected layer should have a learnable bias
#bias_gate specified whether the fully connected layer for the gate should have a learnable bias
    def __init__(self, d_model: int, d_ff: int,
        dropout: float = 0.1,
        activation=nn.ReLU(),
        is_gated: bool = False,
        bias1: bool = True,
        bias2: bool = True,
        bias_gate: bool = True):
        super().__init__()
        #W1 & b1
        self.layer1=nn.Linear(d_model,d_ff,bias=bias1)
        #W2 & b2
        self.layer2=nn.Linear(d_ff,d_model,bias=bias2)
        self.dropout=nn.Dropout(dropout)
        self.activation=activation
        self.is_gated=is_gated
        #the layer that will be multiplied by the gate
        if is_gated:
            self.linear_v=nn.Linear(d_model,d_ff,bias=bias_gate)
    def forward(self, x: torch.Tensor):
        #f(xW1+b1)
        g=self.activation(self.layer1(x))
        #if gated (multiply g by the result of the gate layer)
        if self.is_gated:
            x=g*self.linear_v(x)
        else:
            x=g
        x=self.dropout(x)
        return self.layer2(x)



                