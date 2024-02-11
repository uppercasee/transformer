from torch import nn


class TokenEmbedding(nn.Embedding):
    """
    Token Embedding using torch.nn
    they will dense representation of word using weighted matrix

    Args:
        vocab_size: size of vocabulary
        d_model: dimensions of model

    Returns:
        output: [batch_size, length, d_model]
    """

    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)
