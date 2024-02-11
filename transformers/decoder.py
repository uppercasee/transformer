import torch
from torch import nn

from transformers.embedding.transformer_embedding import TransformerEmbedding
from transformers.layers.decoder_layer import DecoderLayer


class Decoder(nn.Module):
    """
    Decoder module is composed of a stack of N = 6 identical layers.
    each layer has three sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a multi-head attention mechanism, and the third is a simple, position-wise fully connected feed-forward network.

    Args:
        dec_voc_size: decoder vocabulary size
        max_len: max length of sequence
        d_model: dimension of model
        ffn_hidden: hidden size of feed forward network
        n_head: number of head
        n_layers: number of layers
        drop_prob: dropout probability
        device: device type

    Returns:
        output: [batch_size, trg_length, d_model]
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
        Args:
            trg: [batch_size, trg_length]
            enc_output: [batch_size, src_length, d_model]
            src_mask: [batch_size, src_length, src_length]
            trg_mask: [batch_size, trg_length, trg_length]

        Returns:
            output: [batch_size, trg_length, d_model]
        """
        # 1. embedding
        output = self.embedding(trg)

        # 2. decoder layers
        for layer in self.layers:
            output = layer(output, enc_output, src_mask, trg_mask)

        # 3. linear
        output = self.linear(output)

        return output
