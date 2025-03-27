import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

class VideoEmotionNet(nn.Module):
    def __init__(self, num_classes=8):
        super().__init__()
        
        # MobileNetV3 骨干网络
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()  # 移除原始分类头
        
        # 新增空间特征提取层（修复此处）
        self.spatial_feature = nn.Sequential(
            nn.Linear(576, 256),   # mobilenet_v3_small最终输出576通道
            nn.Hardswish(),
            nn.Dropout(0.2)        # 原错误点：缺少括号闭合和逗号
        )
        
        # 时间序列处理（输入维度 256）
        self.temporal = nn.LSTM(
            input_size=256,
            hidden_size=128,
            num_layers=2,
            batch_first=True
        )
        
        # 分类器（修复此处）
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        # 输入维度验证
        assert x.dim() == 5, f"输入需要是5D张量 (B,T,C,H,W)，实际维度：{x.dim()}"
        B, T, C, H, W = x.size()
        assert C == 3, f"输入通道数必须为3，实际：{C}"
        
        # 空间特征提取
        spatial_features = []
        for t in range(T):
            frame = x[:, t, ...]  # (B,3,H,W)
            
            # 骨干网络特征提取
            features = self.backbone.features(frame)  # (B, 576, H', W')
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1))  # (B, 576, 1, 1)
            features = torch.flatten(features, 1)  # (B, 576)
            
            # 空间特征处理
            spatial_feat = self.spatial_feature(features)  # (B,256)
            spatial_features.append(spatial_feat)
        
        # 堆叠时序特征
        spatial_features = torch.stack(spatial_features, dim=1)  # (B,T,256)
        
        # 时序建模
        temporal_out, _ = self.temporal(spatial_features)  # (B,T,128)
        last_output = temporal_out[:, -1, :]  # (B,128)
        
        return self.classifier(last_output)  # (B,num_classes)
