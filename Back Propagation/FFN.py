import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.relu_out = []

    def forward(self, x):
        # 第1个线性层
        x_1 = self.linear1(x)

        # ReLU层
        x_relu = self.relu(x_1)
        self.relu_out = (x_relu == 0)  # 该参数用于ReLU层的反向传播

        # 第2个线性层
        x_2 = self.linear2(x_relu)
        return x_2

    def backward(self, dout):
        # 获取第2个Linear层的参数 权重W_2和偏置b_2
        ret_2 = []
        for param in self.linear2.parameters():
            ret_2.append(param)
        W_2 = ret_2[0].T
        b_2 = ret_2[1]

        # 第2个Linear层的反向传播
        dx_2 = torch.matmul(dout, W_2.T)

        # ReLU层的反向传播
        dx_2[self.relu_out] = 0
        dx_1 = dx_2

        # 获取第1个Linear层的参数 权重W_1和偏置b_1
        ret_1 = []
        for param in self.linear1.parameters():
            ret_1.append(param)
        W_1 = ret_1[0].T
        b_1 = ret_1[1]

        # 第1个Linear层的反向传播
        dx_1 = torch.matmul(dx_1, W_1.T)

        return dx_1


# # 输出梯度值 用于在hook中调用
# def print_grad(grad):
#     print('grad is \n', grad)
#
#
# # FFN验证
# x = torch.randn((2, 3), requires_grad=True)
# print("x value is :", x)
# FFN = PositionwiseFeedForward(3, 4)
# y = FFN.forward(x)
# y.register_hook(print_grad)
# x.register_hook(print_grad)
# y.sum().backward()
#
# dout = torch.ones((2, 3))
# print("dout value is: ", dout)
# dx = FFN.backward(dout)
# print("dx value is: ", dx)
