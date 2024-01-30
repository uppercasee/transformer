import torch
from torch import nn

from transformer.encoder import Encoder
from transformer.decoder import Decoder


class Transformer(nn.Module):
    """
    Transformer is composed of the encoder and decoder. Since the output of the encoder is a sequence of vectors,
    we need to add a linear layer to transform them into a sequence of scores, representing the probability of each word in the target language.

    :param src_pad_idx: index of padding token in source language
    :param trg_pad_idx: index of padding token in target language
    :param trg_sos_idx: index of start of sentence token in target language
    :param enc_voc_size: size of encoder vocabulary
    :param dec_voc_size: size of decoder vocabulary
    :param d_model: dimension of model
    :param ffn_hidden: dimension of feed forward
    :param n_head: number of heads
    :param n_layers: number of layers
    :param drop_prob: dropout rate
    :param device: device type

    :return: [batch_size, trg_length, dec_voc_size]
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
        :param src: [batch_size, src_length]
        :return: [batch_size, 1, 1, src_length]
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
        :param src: [batch_size, src_length]
        :param trg: [batch_size, trg_length]
        :return: [batch_size, trg_length, dec_voc_size]
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
