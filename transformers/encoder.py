from torch import nn

from transformers.embedding.transformer_embedding import TransformerEmbedding
from transformers.layers.encoder_layer import EncoderLayer


class Encoder(nn.Module):
    """
    Encoder module is composed of a stack of N = 6 identical layers.
    each layer has two sub-layers. The first is a multi-head self-attention mechanism,
    and the second is a simple, position-wise fully connected feed-forward network.

    Args:
        enc_voc_size: encoder vocabulary size
        max_len: max length of sequence
        d_model: dimension of model
        ffn_hidden: hidden size of feed forward network
        n_head: number of head
        n_layers: number of layers
        drop_prob: dropout probability
        device: device type

    Returns:
        output: [batch_size, src_length, d_model]
    """

    def __init__(
        self,
        enc_voc_size,
        max_len,
        d_model,
        ffn_hidden,
        n_head,
        n_layers,
        drop_prob,
        device,
    ):
        super(Encoder, self).__init__()
        self.embedding = TransformerEmbedding(
            d_model=d_model,
            max_len=max_len,
            vocab_size=enc_voc_size,
            drop_prob=drop_prob,
            device=device,
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model=d_model,
                    ffn_hidden=ffn_hidden,
                    n_head=n_head,
                    drop_prob=drop_prob,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, src_mask):
        """
        :param x: [batch_size, src_length]
        :param src_mask: [batch_size, src_length]

        :return: [batch_size, src_length, d_model]
        """
        # 1. embedding
        output = self.embedding(x)

        # 2. encoder layers
        for layer in self.layers:
            output = layer(output, src_mask)

        return output
