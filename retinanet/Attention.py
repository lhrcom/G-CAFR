import torch
import torch.nn as nn
import torch.nn.functional as F
from .AdaIN import *
import fvcore.nn.weight_init as weight_init
    
class CrossAttentionBlock(nn.Module):
    def __init__(self, in_channels,num_heads=4):
        super(CrossAttentionBlock, self).__init__()
        
        self.g = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)

        self.W = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0)
        for layer in [self.g, self.theta, self.phi, self.W]:
            weight_init.c2_xavier_fill(layer)
        
        self.adain = AdaIN_block()
        
    def forward(self, x):
        batch_size, C, height, width = x[0].size()

        g_x = self.g(x[0]).view(batch_size, -1, height * width).permute(0, 2, 1)

        theta_x = self.theta(x[1]).view(batch_size, -1, height * width)
        phi_x = self.phi(x[1]).view(batch_size, -1, height * width).permute(0, 2, 1)
        
        f = torch.matmul(phi_x, theta_x)
        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x).view(batch_size, C // 8, height, width)
        
        W_y = self.W(y)
        z = self.adain(x[0], W_y)
        return z

    
class SelfAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttentionBlock, self).__init__()
        # 定义Q, K, V的1x1卷积层，将输入通道数in_channels映射到in_channels // 8
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.key = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)
        self.value = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1, stride=1, padding=0)

        # 定义输出层，将注意力结果in_channels // 8映射回in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.out_conv = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1, stride=1, padding=0)

        for layer in [self.query, self.key, self.value, self.out_conv]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        batch_size, C, height, width = x.size()
        
        # 1. 计算Q, K, V
        proj_query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1) # Q
        proj_key = self.key(x).view(batch_size, -1, height * width) # K
        proj_value = self.value(x).view(batch_size, -1, height * width) # V
        
        # 2. 计算注意力图（Attention Map）
        attention = torch.bmm(proj_query, proj_key) # Q * K^T
        attention = F.softmax(attention, dim=-1)
        
        # 3. 计算注意力结果
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        
        # 4. 加权并添加残差连接
        out = self.out_conv(out)
        out = self.gamma * out + x 
        return out