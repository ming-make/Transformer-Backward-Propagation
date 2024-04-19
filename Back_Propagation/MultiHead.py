import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1]
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)
    if mask is not None:
        scaled += (mask * -1e9)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention


def scaled_dot_product_backward(dout, q, k, v, attention):
    d_k = q.size()[-1]
    x_1 = torch.matmul(dout, v.transpose(-1, -2))
    grad_v = torch.matmul(attention.transpose(-1, -2), dout)
    x_2 = attention * x_1
    sum_x_2 = torch.sum(x_2, dim=-1, keepdim=True)
    x_2 -= attention * sum_x_2
    grad_q = torch.matmul(x_2, k) / math.sqrt(d_k)
    grad_k = (torch.matmul(q.transpose(-1, -2), x_2) / math.sqrt(d_k)).transpose(-1, -2)
    return grad_q, grad_k, grad_v


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        # 分拆后的维度
        self.depth = d_model // num_heads
        # W_q W_k W_v三个矩阵
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        # 合并后经过的全连接层
        self.linear_layer = nn.Linear(d_model, d_model)
        self.seq_len_q = None
        self.cache = []

    def spit_heads(self, x):
        seq_length = x.size()[0]
        x = torch.reshape(x, (seq_length, self.num_heads, self.depth))
        x = x.permute(1, 0, 2)
        return x

    def forward(self, q, k, v, mask=None):
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 拆分为多头
        q = self.spit_heads(q)
        k = self.spit_heads(k)
        v = self.spit_heads(v)
        self.cache.append(q)
        self.cache.append(k)
        self.cache.append(v)

        # 计算注意力
        scaled_attention, attention_weights = scaled_dot_product(q, k, v, mask=None)
        self.cache.append(attention_weights)
        scaled_attention = scaled_attention.permute(1, 0, 2)

        # 合并多头
        self.seq_len_q = scaled_attention.size()[0]
        concat_attention = torch.reshape(scaled_attention, (self.seq_len_q, self.d_model))

        out = self.linear_layer(concat_attention)

        return out, attention_weights

    def backward(self, dout):
        # 获取最后Linear层的参数 权重W_l和偏置b_l
        ret_l = []
        for param in self.linear_layer.parameters():
            ret_l.append(param)
        W_l = ret_l[0].T
        b_l = ret_l[1]

        # 最后Linear层的反向传播
        dx_l = torch.matmul(dout, W_l.T)

        dx_l = torch.reshape(dx_l, (self.seq_len_q, self.num_heads, self.depth))

        dx_l = dx_l.permute(1, 0, 2)

        grad_q, grad_k, grad_v = scaled_dot_product_backward(dx_l, self.cache[0], self.cache[1], self.cache[2], self.cache[3])

        grad_q = grad_q.permute(1, 0, 2)
        grad_k = grad_k.permute(1, 0, 2)
        grad_v = grad_v.permute(1, 0, 2)

        seq_len_grad_q = grad_q.size()[0]
        seq_len_grad_k = grad_k.size()[0]
        seq_len_grad_v = grad_v.size()[0]

        grad_q = torch.reshape(grad_q, (seq_len_grad_q, self.d_model))
        grad_k = torch.reshape(grad_k, (seq_len_grad_k, self.d_model))
        grad_v = torch.reshape(grad_v, (seq_len_grad_v, self.d_model))

        # 获取wq层的参数 权重W_q和偏置b_q
        ret_q = []
        for param in self.wq.parameters():
            ret_q.append(param)
        W_q = ret_q[0].T
        b_q = ret_q[1]

        # wq层的反向传播
        dx_q = torch.matmul(grad_q, W_q.T)

        # 获取wk层的参数 权重W_k和偏置b_k
        ret_k = []
        for param in self.wk.parameters():
            ret_k.append(param)
        W_k = ret_k[0].T
        b_k = ret_k[1]

        # wk层的反向传播
        dx_k = torch.matmul(grad_k, W_k.T)

        # 获取wv层的参数 权重W_v和偏置b_v
        ret_v = []
        for param in self.wv.parameters():
            ret_v.append(param)
        W_v = ret_v[0].T
        b_v = ret_v[1]

        # wv层的反向传播
        dx_v = torch.matmul(grad_v, W_v.T)

        d_final = dx_q + dx_k + dx_v

        return d_final


# # 输出梯度值 用于在hook中调用
# def print_grad(grad):
#     print('grad is \n', grad)
#
#
# # 验证多头注意力机制
# y = torch.randn((2, 4), requires_grad=True)
# print("y value is", y)
# mta = MultiHeadAttention(4, 2)
# output, output_weights = mta.forward(y, y, y)
# print("output value is", output)
# output.register_hook(print_grad)
# y.register_hook(print_grad)
# output.sum().backward()
#
# dout = torch.ones((2, 4))
# dfinal = mta.backward(dout)
# print(dfinal)
