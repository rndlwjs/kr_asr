#한국외대 언어공학연구소
#Conformer 구현
#클라스 모듈명은 논문 표기 참고

import torch
import torch.nn as nn
import torch.nn.functional as F

class feed_forward_module_1(nn.Module):
    def __init__(self, C):
        super(feed_forward_module_1, self).__init__()
    
    def forward(self,x):
        return x

#Relative Position Representations in Transformer
class multi_head_self_attention(nn.Module):
    def __init__(self, C):
        super(multi_head_self_attention, self).__init__()
    
    def forward(self,x):
        return x

class convolutional_module(nn.Module):
    def __init__(self, C):
        super(convolutional_module, self).__init__()
    
    def forward(self,x):
        return x

class feed_forward_module_2(nn.Module):
    def __init__(self, C):
        super(feed_forward_module_2, self).__init__()
    
    def forward(self,x):
        return x

class conformer_block(nn.Module):
    def __init__(self, C):
        super(conformer_block, self).__init__()
        self.feed_forward_module_1      = feed_forward_module_1()
        self.multi_head_self_attention  = multi_head_self_attention()
        self.convolutional_module       = convolutional_module()
        self.feed_forward_module_2      = feed_forward_module_2()

    def forward(self, x):
        residual    = x
        x           = self.feed_forward_module_1(x)             #what is 1/2 *
        x           = x + residual
        #skip connection
        x           = x + self.multi_head_self_attention(x)
        x           = x + self.convolutional_module(x)
        x           = x + self.feed_forward_module_2(x)         #what is 1/2 *
        #layernorm: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
        x           = nn.LayerNorm(x)
        return x