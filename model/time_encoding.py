import torch
import torch.nn as nn
import numpy as np


class TimeEncode(torch.nn.Module):
    # Time Encoding proposed by TGAT
    def __init__(self, dimension):
        super(TimeEncode, self).__init__()

        self.dimension = dimension
        self.w = torch.nn.Linear(1, dimension)

        self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                           .float().reshape(dimension, -1))
        self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

    def forward(self, t):
        # t has shape [batch_size, seq_len]
        # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
        t = t.unsqueeze(dim=2)

        # output has shape [batch_size, seq_len, dimension]
        output = torch.cos(self.w(t))

        return output


def t2v(tau, f, w, b, w0, b0, arg=None):
    v1 = torch.mm(tau, w0) + b0
    if arg:
        v2 = f(torch.mm(tau, w) + b, arg)
    else:
        v2 = f(torch.mm(tau, w) + b)
    return torch.cat([v1, v2], dim=-1)


class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.sin

    def forward(self, tau):
        """

        :param tau: shape (batch_size, input_time_dim) or (batch_size, max_neighbors_num, input_time_dim)
        :return:
        """
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.Parameter(torch.randn(in_features, 1))
        self.w = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.b = nn.Parameter(torch.randn(in_features, out_features - 1))
        self.f = torch.cos

    def forward(self, tau):
        """

        :param tau: shape (batch_size, input_time_dim) or (batch_size, max_neighbors_num, input_time_dim)
        :return:
        """
        return t2v(tau, self.f, self.w, self.b, self.w0, self.b0)


class Time2Vec(nn.Module):
    def __init__(self, activation, embedding_dim):
        super(Time2Vec, self).__init__()
        self.dimension = embedding_dim
        if activation == "sin":
            self.time_encoding = SineActivation(1, embedding_dim)
        elif activation == "cos":
            self.time_encoding = CosineActivation(1, embedding_dim)
        else:
            raise ValueError(f"wrong value for activation {activation}")\

    def forward(self, x):
        """

        :param x: shape (batch_size, 1)
        :return:
        """
        # x shape -> (batch_size, embedding_dim)
        x = self.time_encoding(x)
        return x


class PeriodicTimeEncoder(nn.Module):
    """
    encoder which encodes periodic time to a vector
    """

    def __init__(self, embedding_dim):
        super(PeriodicTimeEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale_factor = (1 / (embedding_dim // 2)) ** 0.5

        self.w = nn.Parameter(torch.randn(1, embedding_dim // 2))
        self.b = nn.Parameter(torch.randn(1, embedding_dim // 2))

    def forward(self, input_relative_time):
        """
        :param input_relative_time: shape (batch_size, input_time_dim) or (batch_size, max_set_size, input_time_dim)
               input_time_dim = 1 since the feature denotes relative time (scalar)
        :return:
            time_encoding, shape (batch_size, embedding_dim) or (batch_size, max_set_size, embedding_dim)
        """

        # cos_encoding, shape (batch_size, embedding_dim // 2) or (batch_size, max_set_size, embedding_dim // 2)
        cos_encoding = torch.cos(torch.matmul(input_relative_time, self.w) + self.b)
        # sin_encoding, shape (batch_size, embedding_dim // 2) or (batch_size, max_set_size, embedding_dim // 2)
        sin_encoding = torch.sin(torch.matmul(input_relative_time, self.w) + self.b)

        # time_encoding, shape (batch_size, embedding_dim) or (batch_size, max_set_size, embedding_dim)
        time_encoding = self.scale_factor * torch.cat([cos_encoding, sin_encoding], dim=-1)

        return time_encoding

# if __name__ == "__main__":
#     sineact = SineActivation(1, 64)
#     cosact = CosineActivation(1, 64)
#     time_enc = Time2Vec("cos",64)
#     # print(sineact(torch.Tensor([[7]])))
#     # print(cosact(torch.Tensor([[7]])))
#     print(time_enc(torch.Tensor([[7]])))