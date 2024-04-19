import torch
import torch.nn as nn

from FFN import PositionwiseFeedForward
from LayerNorm import LayerNormalization
from MultiHead import MultiHeadAttention


class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, num_heads):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
        self.norm1 = LayerNormalization(parameters_shape=[d_model])
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm2 = LayerNormalization(parameters_shape=[d_model])

    def forward(self, x):
        attn_output, _ = self.attention.forward(x, x, x, mask=None)

        out1 = self.norm1.forward(x + attn_output)

        ffn_output = self.ffn.forward(out1)

        out2 = self.norm2.forward(out1 + ffn_output)

        return out2

    def backward(self, dout):
        d_norm2 = self.norm2.backward(dout)
        d_add2 = d_norm2
        d_ffn = self.ffn.backward(d_add2)
        d_ffn_1 = d_ffn + d_add2
        d_norm1 = self.norm1.backward(d_ffn_1)
        d_add1 = d_norm1
        d_multi_head = self.attention.backward(d_norm1)
        d_multi_head_1 = d_multi_head + d_add1
        return d_multi_head_1


# 验证Encoder Layer
sample_encoder_layer = EncoderLayer(4, 4, 2)
input_tensor = torch.randn((4, 4), requires_grad=True)
print("input_tensor value is", input_tensor)
output_tensor = sample_encoder_layer.forward(input_tensor)
print("output_tensor value is", output_tensor)
target = torch.ones((4, 4), requires_grad=True)
cross_entropy_loss = nn.CrossEntropyLoss()
loss = cross_entropy_loss(output_tensor, target)


ret = []  # 用于保存梯度值


# 输出梯度值 用于在hook中调用
def print_grad(grad):
    ret.append(grad)
    print('grad is \n', grad)


# 验证Encoder层
loss.register_hook(print_grad)
output_tensor.register_hook(print_grad)
input_tensor.register_hook(print_grad)
loss.backward()  # 反向传播


dout = ret[1]
grad = sample_encoder_layer.backward(dout)
print("grad is\n", grad)
