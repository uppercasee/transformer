from torch import nn

from transformers.layers.sublayers.layer_norm import LayerNorm
from transformers.layers.sublayers.multi_head_attention import MultiHeadAttention
from transformers.layers.sublayers.position_wise_feed_forward import (
    PositionwiseFeedForward,
)


class EncoderLayer(nn.Module):
    """
    Encoder Layer module is composed of two sub-layers:
    1. Multi-Head Attention (with padding mask) used to compute the attention weights,
    2. Position-wise Feed-Forward Networks (with residual connection and layer normalization) used to apply a fully connected feed forward network to each position.

    :param d_model: dimension of model
    :param ffn_hidden: dimension of feed forward
    :param n_head: number of heads
    :param drop_prob: dropout rate

    :return: [batch_size, length, d_model]
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

    def forward(self, x, src_mask):
        # 1. self attention
        output, _ = self.attention(q=x, k=x, v=x, mask=src_mask)
        output = self.norm1(output + x)
        output = self.dropout1(output)

        # 2. positionwise feed forward
        output = self.ffn(output)
        output = self.norm2(output + x)
        output = self.dropout2(output)

        return output
