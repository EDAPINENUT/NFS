import torch
import numpy as np
import torch.nn as nn

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, relative_error=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.relative_error = relative_error

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff = (((x.reshape(num_examples,-1) - y.reshape(num_examples,-1)).abs())**(self.p)).mean(dim=-1)
        diff = diff**(1/self.p)
        y_abs = y.reshape(num_examples,-1).abs()  if self.relative_error else 1

        if self.reduction:
            if self.size_average:
                return torch.mean(diff/y_abs)
            else:
                return torch.sum(diff/y_abs)

        return diff/y_abs

    def __call__(self, x, y):
        return self.rel(x, y)
