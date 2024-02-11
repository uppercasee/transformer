import torch
from torch import nn

from transformers.encoder import Encoder
from transformers.decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer is composed of the encoder and decoder. Since the output of the encoder is a sequence of vectors,
    we need to add a linear layer to transform them into a sequence of scores, representing the probability of each word in the target language.

    Args:
        src_pad_idx: source padding index
        trg_pad_idx: target padding index
        trg_sos_idx: target start of sentence index
        enc_voc_size: encoder vocabulary size
        dec_voc_size: decoder vocabulary size
        d_model: dimension of model
        n_head: number of head
        max_len: max length of sequence
        ffn_hidden: hidden size of feed forward network
        n_layers: number of layers
        drop_prob: dropout probability
        device: device type

    Returns:
        output: [batch_size, trg_length, dec_voc_size]
    """

    def __init__(
        self,
        src_pad_idx,
        trg_pad_idx,
        trg_sos_idx,
        enc_voc_size,
        dec_voc_size,
        d_model,
        n_head,
        max_len,
        ffn_hidden,
        n_layers,
        drop_prob,
        device,
    ):
        super().__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device
        self.encoder = Encoder(
            enc_voc_size=enc_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device,
        )

        self.decoder = Decoder(
            dec_voc_size=dec_voc_size,
            max_len=max_len,
            d_model=d_model,
            ffn_hidden=ffn_hidden,
            n_head=n_head,
            n_layers=n_layers,
            drop_prob=drop_prob,
            device=device,
        )

        self.linear = nn.Linear(d_model, dec_voc_size)
        nn.init.xavier_uniform_(
            self.linear.weight
        )  # Xavier/Glorot initialization method
        # nn.init.normal_(self.linear.weight, mean=0, std=0.1) # Normal distribution initialization method

    def make_src_mask(self, src):
        """
        make source mask

        Args:
            src: [batch_size, src_length]

        Returns:
            src_mask: [batch_size, 1, 1, src_length]
        """
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        """
        make target mask
        :param trg: [batch_size, trg_length]
        :return: [batch_size, 1, trg_length, trg_length]
        """
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        """
        Args:
            src: [batch_size, src_length]
            trg: [batch_size, trg_length]

        Returns:
            output: [batch_size, trg_length, dec_voc_size]
        """
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)

        # 1. encoder
        enc_output = self.encoder(src, src_mask)

        # 2. decoder
        dec_output = self.decoder(trg, enc_output, src_mask, trg_mask)

        # 3. linear
        output = self.linear(dec_output)

        return output
