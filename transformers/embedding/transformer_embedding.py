from torch import nn

from transformers.embedding.positional_encoding import PositionalEncoding
from transformers.embedding.token_embedding import TokenEmbedding


class TransformerEmbedding(nn.Module):
    """
    Transformer Embedding using torch.nn
    they will dense representation of word using weighted matrix with positional encoding information

    positional encoding can give positional information to network

    Args:
        vocab_size: size of vocabulary
        d_model: dimensions of model
        max_len: max length of sequence
        drop_prob: dropout probability
        device: device type

    Returns:
        output: [batch_size, length, d_model]
    """

    def __init__(self, vocab_size, d_model, max_len, drop_prob, device):
        """
        class for word embedding that included positional information

        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len, device)
        self.drop_out = nn.Dropout(p=drop_prob)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb + pos_emb)
