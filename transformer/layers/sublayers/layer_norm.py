import torch
from torch import nn


class LayerNorm(nn.Module):
    """
    Layer Normalization module is used to normalize the outputs of each sub-layer.
    it is applied to the input of each sub-layer, before it is passed through the sub-layer itself.

    :param d_model: dimension of model
    :param eps: epsilon value to avoid zero division

    :return: [batch_size, length, d_model]
    """

    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # 1. get mean and std
        # x: [batch_size, length, d_model]
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        # 2. normalize
        out = (x - mean) / torch.sqrt(var + self.eps)

        # 3. scale and shift
        out = self.gamma * out + self.beta

        return out
