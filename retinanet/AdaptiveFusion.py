# import torch
# import torch.nn as nn
# from .Attention import CrossAttentionBlock
# from .mamba import SimpleBiMambaBlock  # 假设你保存 Mamba 版本在这个文件

# class AdaptiveFusion(nn.Module):
#     """
#     根据特征层分辨率选择融合方式：
#     - 底层/中层 -> SimpleBiMambaBlock (轻量快速)
#     - 高层 -> CrossAttentionBlock (精细对齐)
#     """
#     def __init__(self, feature_channels):
#         """
#         feature_channels: list[int] 每个特征层的通道数, 例如 [256, 512, 1024, 2048]
#         """
#         super(AdaptiveFusion, self).__init__()
#         self.num_levels = len(feature_channels)
#         self.fusion_blocks = nn.ModuleList()

#         for i, C in enumerate(feature_channels):
#             if i < self.num_levels - 2:
#                 # 底层和中层 -> Mamba
#                 self.fusion_blocks.append(SimpleBiMambaBlock(C))
#             else:
#                 # 高层 -> CrossAttention
#                 self.fusion_blocks.append(CrossAttentionBlock(C))

#     def forward(self, features_rgb, features_evt):
#         """
#         features_rgb, features_evt: list[Tensor], 每个Tensor对应一层特征
#             Tensor shape: (B, C, H, W)
#         """
#         fused_features = []
#         for i, fusion_block in enumerate(self.fusion_blocks):
#             fused = fusion_block((features_rgb[i], features_evt[i]))
#             fused_features.append(fused)
#         return fused_features
