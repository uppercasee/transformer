from torch import nn


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Networks module is used to inject some non-linearity into the model.
    It is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    Dropout is also applied before the second linear transformation.
    It can be described as FFN(x) = max(0, xW1 + b1)W2 + b2.

    Args:
        d_model: dimension of model
        hidden: hidden size of feed forward network
        drop_prob: dropout probability

    Returns:
        output: [batch_size, seq_length, d_model]
    """

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
