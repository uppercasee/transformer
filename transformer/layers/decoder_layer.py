from torch import nn

from transformer.layers.sublayers.layer_norm import LayerNorm
from transformer.layers.sublayers.multi_head_attention import MultiHeadAttention
from transformer.layers.sublayers.position_wise_feed_forward import (
    PositionwiseFeedForward,
)


class DecoderLayer:
    """
    Decoder Layer module is composed of three sub-layers:
    1. Masked Multi-Head Attention (with padding mask and look ahead mask) used to compute the attention weights,
    2. Multi-Head Attention (with padding mask) used to compute the attention weights,
    3. Position-wise Feed-Forward Networks (with residual connection and layer normalization) used to apply a fully connected feed forward network to each position.

    :param d_model: dimension of model
    :param ffn_hidden: dimension of feed forward
    :param n_head: number of heads
    :param drop_prob: dropout rate

    :return: [batch_size, length, d_model]
    """

    def __init__(self, d_model, ffn_hidden, n_head, drop_prob):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.ffn = PositionwiseFeedForward(
            d_model=d_model, hidden=ffn_hidden, drop_prob=drop_prob
        )
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=drop_prob)

    def forward(self, dec_input, enc_output, src_mask, trg_mask):
        """
        :param dec_input: [batch_size, trg_length, d_model]
        :param enc_output: [batch_size, src_length, d_model]
        :param src_mask: [batch_size, src_length]

        :return: [batch_size, trg_length, d_model], [batch_size, head, trg_length, src_length]
        """
        # 1. masked self attention
        output, _ = self.self_attention(
            q=dec_input, k=dec_input, v=dec_input, mask=trg_mask
        )
        output = self.norm1(output + dec_input)
        output = self.dropout1(output)

        # 2. encoder-decoder attention
        output, _ = self.enc_dec_attention(
            q=output, k=enc_output, v=enc_output, mask=src_mask
        )
        output = self.norm2(output + dec_input)
        output = self.dropout2(output)

        # 3. positionwise feed forward
        output = self.ffn(output)
        output = self.norm3(output + dec_input)
        output = self.dropout3(output)

        return output

    # def forward(self, dec_input, enc_output, src_mask, trg_mask):
    #     """
    #     :param dec_output: decoder input
    #     :param enc_input: encoder output
    #     :param src_mask: source mask
    #     :param trg_mask: target mask
    #     :return: decoder output
    #     """
    #     # 1. compute self attention
    #     _x = dec_input
    #     x = self.self_attention(q=dec_input, k=dec_input, v=dec_input, mask=trg_mask)
    #     x = self.dropout1(x)
    #     x = self.norm1(x + _x)

    #     # 2. compute encoder-decoder attention
    #     _x = x
    #     x = self.enc_dec_attention(q=x, k=enc_output, v=enc_output, mask=src_mask)
    #     x = self.dropout2(x)
    #     x = self.norm2(x + _x)

    #     # 3. positionwise feed forward network
    #     _x = x
    #     x = self.ffn(x)
    #     x = self.dropout3(x)
    #     x = self.norm3(x + _x)
    #     return x
