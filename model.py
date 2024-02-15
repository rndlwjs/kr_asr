#한국외대 언어공학연구소
#Conformer 구현
#클라스 모듈명은 논문 표기 참고

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy

class feed_forward_module_1(nn.Module):
    def __init__(self, DIM):
        super(feed_forward_module_1, self).__init__()
        self.layernorm      = nn.LayerNorm(DIM)
        self.linear1        = nn.Linear(DIM, DIM*4)     #expansion factor 4
        self.swish          = nn.SiLU(DIM)
        self.dropout        = nn.Dropout(p=0.1)         #probability is 0.1
        self.linear2        = nn.Linear(DIM*4, DIM)     #project back

    def forward(self,x):
        residual    = x
        x           = self.layernorm(x)
        x           = self.linear1(x)
        x           = self.swish(x)
        x           = self.dropout(x)
        x           = self.linear2(x)
        x           = self.dropout(x)
        x           += residual

        return x

class scale_dot_attention(nn.Module):
    def __init__(self, DIM):
        super(scale_dot_attention, self).__init__()
        self.q              = nn.Linear(DIM, DIM)
        self.k              = nn.Linear(DIM, DIM)
        self.v              = nn.Linear(DIM, DIM)
        self.softmax        = nn.Softmax(dim=1) #Q4 How to choose the dimension for softmax?
        self.dim            = DIM

    def forward(self, x):
        Q           = self.q(x)
        K           = self.k(x)
        V           = self.v(x)
        scale       = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(self.dim)
        output      = torch.matmul(self.softmax(scale), V)
        return output

#Relative Position Representations in Transformer
class multi_head_self_attention(nn.Module):
    def __init__(self, DIM, num_Head=4):
        super(multi_head_self_attention, self).__init__()
        self.layernorm              = nn.LayerNorm1d(DIM)
        self.scale_dot_attention    = scale_dot_attention(DIM)
        self.multi  = [for i in num_head: self.scale_dot_attention]
        self.multi_head             = []
        #for i in range(num_Head): self.multi_head.append(copy.deepcopy(self.scale_dot_attention))
        #self.multi  = nn.ModuleList(self.multi_head)

        self.num_head               = num_Head
        self.linear                 = nn.Linear(DIM, DIM)
        self.dropout                = nn.Dropout(p=0.1)

    def forward(self,x):
        x   = self.layernorm(x)
        #Multi-head Attention
        atts = []
        for att in range(self.num_head):
            atts.append(self.multi[att](x))
        
#        x   = torch.cat([i for i in self.num_head: self.scale_dot_attention(x)], dim=1) #cat dim=1?

        x   = self.linear(x)
        x   = self.dropout(x)
        return x

class convolutional_module(nn.Module):
    def __init__(self, DIM):
        super(convolutional_module, self).__init__()
        self.layernorm      = nn.LayerNorm(DIM)
        self.pointwise1     = nn.Conv1d(DIM, DIM*2, kernel_size=1)     #expansion factor of 2 projecting the number of channel with GLU layer
        self.glu            = nn.GLU(dim=1)
        self.depthwise      = nn.Conv1d(DIM, DIM, kernel_size=31, padding='same', groups=DIM)   #kernel_size should be odd number
        self.bn             = nn.BatchNorm1d(DIM)
        self.swish          = nn.SiLU(DIM)
        self.pointwise2     = nn.Conv1d(DIM, DIM, kernel_size=1)
        self.dropout        = nn.Dropout(p=0.1)

    def forward(self,x):
        residual        = x
        x               = self.pointwise1(x)
        x               = self.glu(x)
        x               = self.depthwise(x)
        x               = self.bn(x)
        x               = self.swish(x)
        x               = self.pointwise2(x)
        x               = self.dropout(x)
        x               += residual
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
