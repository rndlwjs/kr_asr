import torch
from model import convolutional_module
from model import feed_forward_module_1

"""input = torch.randn(1, 144, 80)
conv = convolutional_module(DIM=144)

print(conv(input).shape)

input = input.transpose(1, 2)       #(batch, freq, time)
ffm = feed_forward_module_1(DIM=144)

print(ffm(input).shape)"""

from model import scale_dot_attention

input = torch.randn(1, 80, 144)
s_attn = scale_dot_attention(DIM=144)
print(s_attn(input).shape)