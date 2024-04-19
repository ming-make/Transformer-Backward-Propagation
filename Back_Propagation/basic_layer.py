import numpy as np

d_k = d_v = 64  # K(=Q), V的维度


def print_grad(grad):
    print('grad is \n', grad)


class AddLayer:  # 加法
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy


class MatMul:  # 矩阵乘法
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx


# 矩阵加法和矩阵乘法的验证
# x = torch.rand(2, 1, requires_grad=True)
# print('x value is \n', x)
# some = torch.rand(2, 1, requires_grad=True)
# print('some value is \n', some)
# y = x + some
# print('y value is \n', y)
# z = torch.rand(1, 3, requires_grad=True)
# print(z)
# q = y * z
# print('q value is \n', q)
# lr = 1e-3
#
# q.register_hook(print_grad)
# q.sum().backward()  # 梯度求解
# print('z grad is \n', z.grad)
# print('x grad is \n', x.grad)
# print('some grad is \n', some.grad)
#
# add = AddLayer()
# y_1 = add.forward(x, some)
# mul = MatMul(z)
# q_1 = mul.forward(y_1)
# dout = torch.ones(size=(2, 3))
# print(mul.backward(dout))
# x_grad, some_grad = add.backward(mul.backward(dout))
# print('x grad is \n', x_grad)
# print('some grad is \n', some_grad)

def softmax(x):  # softmax函数
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)  # 溢出对策
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x) / np.sum(np.exp(x))

    return x


class Softmax:  # softmax层
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        ret = []
        for i in range(dout.shape[0]):
            softmax_grad = np.diag(self.out[i]) - np.outer(self.out[i], self.out[i])
            ret.append(np.dot(softmax_grad, dout[i].T))
        ret = np.array(ret)
        return ret

# # Softmax验证
# src = torch.randn(2, 3, requires_grad=True)
# print('src value is \n', src)
# y = F.softmax(src, dim=-1)
# print('y value is \n', y)
# y.register_hook(print_grad)
# y.sum().backward()
# print('src grad is \n', src.grad)
#
#
# src_1 =src.detach().numpy().round(4)
# print('src_1 value is \n', src_1)
# soft = Softmax()
# y_1 = soft.forward(src_1).round(4)
# print('y_1 value is \n', y_1)
# dout = np.ones((2, 3)).round(1)
# ret = soft.backward(dout).round(1)
# print('src_1 grad is \n', ret)


class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

# # ReLu验证
# x = torch.randn((2, 3), requires_grad=True)
# print('x value is \n', x)
# m = torch.nn.ReLU()
# y = m(x)
# print('y value is \n', y)
# y.register_hook(print_grad)
# y.sum().backward()
# print('x grad is \n', x.grad)
#
# x_1 = x.detach().numpy().round(4)
# print('x_1 value is \n', x_1)
# relu = Relu()
# y_1 = relu.forward(x_1).round(4)
# print('y_1 value is \n', y_1)
# dout = np.ones((2, 3)).round(1)
# x_1_grad = relu.backward(dout)
# print('x_1 grad is \n', x_1_grad)


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b

        self.x = None
        self.original_x_shape = None
        # 权重和偏置参数的导数
        self.dW = None
        self.db = None

    def forward(self, x):
        # 对应张量
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)

        dx = dx.reshape(*self.original_x_shape)  # 还原输入数据的形状（对应张量）
        return dx

# # 全连接层验证
# x = torch.randn((2, 3), requires_grad=True)
# print('x value is \n', x)
# embedding = nn.Linear(3, 4)
# # 查看模型参数
# ret = []
# for param in embedding.parameters():
#     print(param)
#     ret.append(param)
# W = ret[0].detach().numpy().round(4).T
# b = ret[1].detach().numpy().round(4)
# y = embedding(x)
# print('y value is \n', y)
# y.register_hook(print_grad)
# y.sum().backward()
# print('x grad is \n', x.grad)
#
# aff = Affine(W, b)
# x_1 = x.detach().numpy().round(4)
# print('x_1 value is \n', x_1)
# y_1 = aff.forward(x_1).round(4)
# print('y_1 value is \n', y_1)
# dout = np.ones((2, 4))
# x_1_grad = aff.backward(dout).round(4)
# print('x_1 grad is \n', x_1_grad)
