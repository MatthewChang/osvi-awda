import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math
import numpy as np

class PositionalEncoding(nn.Module):
    """
    PositionalEncoding from Pytorch Seq2Seq Documentation
    source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    ^ slightly modified
    
    why sum instead of cat? see here: https://github.com/tensorflow/tensor2tensor/issues/1591
    and: https://www.reddit.com/r/MachineLearning/comments/cttefo/d_positional_encoding_in_transformer/exs7d08/?utm_source=reddit&utm_medium=web2x&context=3

    works out of the box for inputs with shape (B,T,D)
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000,time_dim=1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.time_dim = 1

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(self.time_dim)]
        return self.dropout(x)


class TemporalPositionalEncoding(nn.Module):
    """
    Modified PositionalEncoding from Pytorch Seq2Seq Documentation
    source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(1, 2)
        self.register_buffer('pe', pe)

    def forward(self, x):
        assert len(x.shape) >= 3, "x requires at least 3 dims! (B, C, ..)"
        old_shape = x.shape
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = x + self.pe[:,:,:x.shape[-1]]
        return self.dropout(x).reshape(old_shape)
