import torch
import torch.nn as nn

class IndexSampler(nn.Module):
    def __init__(self, input_size, idxs):
        super(IndexSampler, self).__init__()
        self.idxs = idxs
      
    def forward(self, x):
        subsample = torch.zeros_like(x)
        for i in self.idxs:
            subsample[:, i] = x[:, i]
        return subsample