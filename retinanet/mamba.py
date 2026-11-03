import torch
import torch.nn as nn
from .AdaIN import *
import fvcore.nn.weight_init as weight_init
from mamba_ssm import Mamba  # pip install mamba-ssm

class SimpleBiMambaBlock(nn.Module):
    def __init__(self, in_channels):
        super(SimpleBiMambaBlock, self).__init__()
        self.in_channels = in_channels

        # 特征投影 (降维 -> 保持计算高效)
        self.proj_f = nn.Conv2d(in_channels, in_channels // 2, 1)
        self.proj_e = nn.Conv2d(in_channels, in_channels // 2, 1)

        # 输出恢复
        self.out_f = nn.Conv2d(in_channels // 2, in_channels, 1)
        self.out_e = nn.Conv2d(in_channels // 2, in_channels, 1)

        # AdaIN 融合
        self.adain = AdaIN_block()

        # Mamba selective scan
        self.mamba = Mamba(
            d_model=in_channels*2,  # 输入维度
            d_state=16,           # 状态维度
            d_conv=4,             # 卷积核大小
            expand=2
        )

        # 初始化
        for layer in [self.proj_f, self.proj_e, self.out_f, self.out_e]:
            weight_init.c2_xavier_fill(layer)

    def forward(self, x):
        """
        x: tuple(Ff, Fe)，两个模态特征 (B, C, H, W)
        """
        Ff, Fe = x
        B, C, H, W = Ff.size()

        # 降维投影
        Ff_p = self.proj_f(Ff)   # (B, C/2, H, W)
        Fe_p = self.proj_e(Fe)

        # 互相引导 (用 AdaIN 而不是 QK)
        Ff_aligned = self.adain(Ff_p, Fe_p)  # Fe -> Ff
        Fe_aligned = self.adain(Fe_p, Ff_p)  # Ff -> Fe

        # 恢复通道
        Ff_out = self.out_f(Ff_aligned)  # (B, C, H, W)
        Fe_out = self.out_e(Fe_aligned)

        # 拼接两个模态
        fused = torch.cat([Ff_out, Fe_out], dim=1)  # (B, 2C, H, W)

        # 转为序列输入 Mamba
        fused_seq = fused.flatten(2).transpose(1, 2)  # (B, HW, 2C)
        fused_seq = self.mamba(fused_seq)             # (B, HW, 2C)

        # 还原 (B, 2C, H, W)
        fused_out = fused_seq.transpose(1, 2).view(B, 2*C, H, W)
        return fused_out
