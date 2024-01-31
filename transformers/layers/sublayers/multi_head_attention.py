from torch import nn
from transformers.layers.sublayers.scale_dot_product_attention import (
    ScaleDotProductAttention,
)

# from scale_dot_product_attention import ScaleDotProductAttention


class MultiHeadAttention(nn.Module):
    """
    Multi Head Attention module

    1. W_q, W_k, W_v (linear projection)
    2. split tensor by number of heads
    3. do scale dot product attention to compute similarity
    4. concat and pass to linear layer

    Args:
        d_model: dimension of model
        n_head: number of head

    Returns:
        output: [batch_size, length, d_model]
        score: [batch_size, head, length, length]
    """

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()

        # 1. W_q, W_k, W_v (linear projection)
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # 2. W_o
        self.W_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.W_q(q), self.W_k(k), self.W_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product attention to compute similarity
        # q, k, v: [batch_size, head, length, d_tensor]
        # score: [batch_size, head, length, length]
        q, score = self.attention(q, k, v, mask)

        # 4. concat and pass to linear layer
        # q: [batch_size, head, length, d_tensor]
        # output: [batch_size, length, d_model]
        output = self.concat(q)
        output = self.W_concat(output)

        # TODO : 5. visualize attention map
        # score: [batch_size, head, length, length]
        # score = score.squeeze(1).cpu().data.numpy()
        # score = np.mean(score, axis=0)
        # print(score.shape)
        # plt.imshow(score)
        # plt.show()

        return q, score

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_tensor = tensor.size()
        d_head = d_tensor // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_head).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)
        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_head = tensor.size()
        d_tensor = head * d_head
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_tensor)
        return tensor


if __name__ == "__main__":
    import torch
    from torch.autograd import Variable

    batch_size = 8
    length = 10
    d_model = 512
    n_head = 8

    q = Variable(torch.randn(batch_size, length, d_model))
    k = Variable(torch.randn(batch_size, length, d_model))
    v = Variable(torch.randn(batch_size, length, d_model))

    mask = Variable(torch.ones(batch_size, length, length))

    attention = MultiHeadAttention(d_model, n_head)
    output, score = attention(q, k, v, mask)
    print(output.size())
    print(score.size())
