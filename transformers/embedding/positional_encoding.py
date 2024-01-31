import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """
    Positional Encoding module injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension d_model as the embeddings, so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.

    Args:
        max_len: max length of sequence
        d_model: dimension of model
        device: device type

    Returns:
        output: [max_len, d_model]
    """

    def __init__(self, max_len=512, d_model=512, device="cpu"):
        super(PositionalEncoding, self).__init__()

        # 1. get positional encoding
        # [max_len = 512, d_model = 512]
        self.encoding = torch.zeros(512, 512)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        # 2. get position
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        # 3. get dimension
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        # 4. compute positional encoding
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]
