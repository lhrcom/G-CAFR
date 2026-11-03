import torch
import torch.nn as nn

from .Attention import *

# 在 fusion.py 中新增以下类
class GateNetwork(nn.Module):
    def __init__(self, in_channels):
        super(GateNetwork, self).__init__()
        # # 使用1x1卷积来预测每个通道的权重
        # self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        # # 使用 Sigmoid 激活函数将权重限制在 [0, 1] 范围内
        # self.sigmoid = nn.Sigmoid()
        # self.conv = nn.Conv2d(in_channels, 2, kernel_size=1,stride=1,padding=0)
        self.conv = nn.Sequential(
    nn.Conv2d(in_channels, in_channels//4, kernel_size=1),  # 先降维
    nn.ReLU(inplace=True),
    nn.Conv2d(in_channels//4, 2, kernel_size=1)  # 再输出2个权重通道
)
        self.log_sigma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # return self.sigmoid(self.conv(x))
        logits = self.conv(x)  # [B,2,H,W]
        noise = torch.randn_like(logits) * self.log_sigma.exp()
        gates = self.softmax(logits + noise)  # [B,2,H,W]
        return gates  # gate[:,0] = RGB权重, gate[:,1] = Event权重

class REFusion(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(REFusion, self).__init__()
    
        # 新增：自注意力模块
        self.rgb_self_attention = SelfAttentionBlock(in_planes)
        self.event_self_attention = SelfAttentionBlock(in_planes)

        self.rgb_cross_attention = CrossAttentionBlock(in_planes)
        self.event_cross_attention = CrossAttentionBlock(in_planes)
        

        self.conv0_rgb = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)
        self.conv0_evt = nn.Conv2d(out_planes, out_planes, kernel_size=1, stride=1, padding=0)

        # # 新增：定义门控网络
        # self.gate_rgb = GateNetwork(out_planes)
        # self.gate_evt = GateNetwork(out_planes)
        self.gate = GateNetwork(out_planes*2)

    
    def forward(self, rgb, evt):

        #  # 自注意力
        # rgb_self_att = self.rgb_self_attention(rgb)
        # event_self_att = self.event_self_attention(evt)

        rgb0 = self.conv0_rgb(rgb)
        evt0 = self.conv0_evt(evt)

        # rgb0 = self.conv0_rgb(rgb_self_att)
        # evt0 = self.conv0_evt(event_self_att)
        
       
       
        fused_input = torch.cat([rgb0,evt0],dim=1)
        gates = self.gate(fused_input)
        gate_rgb,gate_evt = gates[:,0:1,:,:],gates[:,1:2,:,:]
        fused = gate_rgb*rgb0+gate_evt*evt0
          
        # mul = rgb0.mul(evt0)
        # rgb1 = rgb0 + mul
        # evt1 = evt0 + mul

     
        # rgb_y = self.rgb_cross_attention([rgb1, evt1])
        # event_y = self.event_cross_attention([evt1, rgb1])

        rgb_y = self.rgb_cross_attention([fused, evt0])
        event_y = self.event_cross_attention([fused, rgb0])


        out = torch.cat([rgb_y, event_y], dim=1)
     
        return out
