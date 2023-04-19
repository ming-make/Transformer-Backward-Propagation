import torch
from torch import nn


class LayerNormalization(nn.Module):

    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape = parameters_shape
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta = nn.Parameter(torch.zeros(parameters_shape))
        self.cache = []

    def forward(self, inputs):
        mean = torch.mean(inputs, dim=-1)
        mean = torch.reshape(mean, (inputs.size()[0], 1))
        var = ((inputs - mean) ** 2)
        var = torch.mean(var, dim=-1)
        var = torch.reshape(var, (inputs.size()[0], 1))
        std = torch.sqrt(var + self.eps)
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        self.cache = (inputs, inputs - mean, std)
        return out

    def backward(self, dout):
        (inputs, inputs_minus_mean, std) = self.cache
        dgamma = torch.sum(dout * inputs_minus_mean / std, dim=-1, keepdim=True)
        dbeta = torch.sum(dout, dim=-1, keepdim=True)

        dlxhat = dout * self.gamma
        dxhatx = 1 / std
        dlvar = -0.5 * torch.sum(self.gamma * inputs_minus_mean * std ** (-3) * dout, dim=1, keepdim=True)
        dlvarx = 2 * inputs_minus_mean / inputs.size()[1]
        dlmu = -1. * torch.sum(dlxhat / std, dim=1, keepdim=True) - 2. * torch.sum(dlvar * inputs_minus_mean, dim=1,
                                                                                   keepdim=True) / inputs.size()[1]
        dx = dlxhat * dxhatx + dlvar * dlvarx + dlmu / inputs.size()[1]

        return dx


# # 输出梯度值 用于在hook中调用
# def print_grad(grad):
#     print('grad is \n', grad)
#
#
# y = torch.randn(2, 3, requires_grad=True)
# print("y value is", y)
#
# layer = nn.LayerNorm(3)
# out = layer(y)
# print("out value is", out)
# out.register_hook(print_grad)
# y.register_hook(print_grad)
# out.sum().backward()
#
# layernorm = LayerNormalization(y.size())
# out = layernorm.forward(y)
# print("out value is", out)
# dout = torch.ones((2, 3))
# dx = layernorm.backward(dout)
# print("dx value is", dx)
