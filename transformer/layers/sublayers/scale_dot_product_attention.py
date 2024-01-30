import math
from torch import nn


class ScaleDotProductAttention(nn.Module):
    """
    Scale Dot Product Attention module

    1. dot product Query with Key^T to compute similarity
    2. apply masking (opt)
    3. pass them softmax to make [0, 1] range
    4. multiply with Value

    :param q: [batch_size, head, length, d_tensor]
    :param k: [batch_size, head, length, d_tensor]
    :param v: [batch_size, head, length, d_tensor]
    :param mask: [batch_size, length, length]
    :return: [batch_size, head, length, d_tensor], [batch_size, head, length, length]
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        # 0. get dimension info
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        # d_tensor = q.size(-1)
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score
