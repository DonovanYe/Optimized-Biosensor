# Adapted from https://github.com/cagladbahadir/LOUPE/

import torch
import torch.nn as nn

common_dtype = torch.float32

# Turn a set of weights to probabilities from 0 to 1
class ProbMask(nn.Module):
    def __init__(self, input_size, slope=5):
        super(ProbMask, self).__init__()

        self.slope = nn.Parameter(torch.tensor(slope, dtype=common_dtype), requires_grad=False)
        weights = self._generate_logits(input_size)
        self.weights = nn.Parameter(weights, requires_grad=True)

    # This generation helps avoid a lot of weights on the edges of sigmoid
    # As per above github link
    def _generate_logits(self, input_size, eps=0.01):
        min = eps
        max = 1 - eps
        uniform = torch.rand(input_size, dtype=common_dtype) *  (max - min) + min
        return - torch.log(1. / uniform - 1.) / self.slope

    def forward(self, x):
      logit_weights = 0 * x[..., 0:1] + self.weights
      return torch.sigmoid(self.slope * logit_weights)

# Rescale probability so mean is {sparsity}
# See Bahadir for more details
class RescaleProbMap(nn.Module):
    def __init__(self, sparsity, **kwargs):
        super(RescaleProbMap, self).__init__(**kwargs)
        self.sparsity = nn.Parameter(torch.tensor(sparsity, dtype=common_dtype), requires_grad=False)

    def forward(self, x):
        xbar = torch.mean(x)
        r = self.sparsity / xbar
        beta = (1 - self.sparsity) / (1 - xbar)
        
        le = torch.le(r, 1).float()
        return le * x * r + (1 - le) * (1 - (1 - x) * beta)

# Uniform mask
class RandomMask(nn.Module):
    def __init__(self, **kwargs):
        super(RandomMask, self).__init__(**kwargs)

    def forward(self, x):
        input_shape = x.size()
        threshs = torch.rand(input_shape, dtype=common_dtype)
        if torch.cuda.is_available():
            threshs = threshs.cuda()
        return 0 * x + threshs

# Binary mask of whether to realize a value with probability p
# Made differentiable with sigmoid
class ThresholdRandomMask(nn.Module):
    def __init__(self, slope = 200, **kwargs):
        super(ThresholdRandomMask, self).__init__(**kwargs)
        self.slope = nn.Parameter(torch.tensor(slope, dtype=common_dtype), requires_grad=False)

    def forward(self, inputs, thresh):
        return torch.sigmoid(self.slope * (inputs - thresh))

# Multiply mask with original data          
class UnderSample(nn.Module):
    def __init__(self, **kwargs):
        super(UnderSample, self).__init__(**kwargs)

    def forward(self, a, b):
        undersample = a * b
        return undersample

from torch.autograd import Function
class Binarize(Function):
    def __init__(self):
        super().__init__()
        
    @staticmethod
    def forward(ctx, rescaled_prob, sparsity):
        input_shape = rescaled_prob.size()
        u = torch.rand(input_shape)
        if torch.cuda.is_available():
            u = u.cuda()
        while abs(torch.mean(torch.le(u, rescaled_prob).float()) - sparsity) / sparsity > 0.1:
            u = torch.rand(input_shape)
            if torch.cuda.is_available():
                u = u.cuda()
        ctx.save_for_backward(rescaled_prob, u)
        mask = torch.le(u, rescaled_prob).float()
        return mask
        
    @staticmethod
    def backward(ctx, grad):
        slope = 10
        rescaled_prob, u = ctx.saved_tensors
        x = rescaled_prob - u
        grad *= slope * torch.exp(-slope * x) / torch.pow(torch.exp(-slope * x) + 1, 2)
        return grad, None
    

class LOUPESampler(nn.Module):
    def __init__(self, input_size, budget):
        super(LOUPESampler, self).__init__()
        self.sparsity = budget / input_size
        self.prob_mask = ProbMask(input_size)
        self.rescale = RescaleProbMap(self.sparsity)
        self.binarize = Binarize.apply
      
    def forward(self, x):
        prob_mask = self.prob_mask(x)
        rescaled = self.rescale(prob_mask)
        binary_map = self.binarize(rescaled, self.sparsity)
        return binary_map * x

        # print(torch.sum(rescaled))
        # print(rescaled)
        # uniform_thresh = self.rand_mask(rescaled)
        # thresholded = self.thresh(rescaled, uniform_thresh)
        # print(thresholded)
        # undersample = self.undersample(x, thresholded)
        # print(thresholded)
        raise
        return undersample