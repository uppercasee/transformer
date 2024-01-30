import torch
from torch import nn

from transformers.embedding.transformer_embedding import TransformerEmbedding
from transformers.layers.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Decoder module is composed of a stack of N = 6 identical layers.
    each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, and the third is a simple, position-wise fully connected feed-forward network.

    :param dec_voc_size: size of decoder vocabulary
    :param max_len: maximum length of sequence
    :param d_model: dimension of model
    :param ffn_hidden: dimension of feed forward
    :param n_head: number of heads
    :param n_layers: number of layers
    :param drop_prob: dropout rate
    :param device: device type

    :return: [batch_size, length, d_model]
    """

    def __init__(
        self,
        dec_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            d_model=d_model,
            drop_prob=drop_prob,
            max_len=max_len,
            vocab_size=dec_voc_size,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

        self.linear = nn.Linear(d_model, dec_voc_size)

    def forward(self, trg, enc_output, src_mask, trg_mask):
        """
        :param trg: [batch_size, trg_length]
        :param enc_output: [batch_size, src_length, d_model]
        :param src_mask: [batch_size, src_length]
        :param trg_mask: [batch_size, trg_length]

        :return: [batch_size, trg_length, d_model]
        """
        # 1. embedding
        output = self.embedding(trg)

        # 2. decoder layers
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, trg_mask)

        # 3. linear
        output = self.linear(output)

        return output
