import torch
import torch.nn as nn

class SimpleFusion(nn.Module):
    def __init__(self, video_model, audio_model):
        super().__init__()
        # 保持单模态模型结构不变
        self.video_net = video_model
        self.audio_net = audio_model
        
        # 冻结参数
        for param in self.video_net.parameters():
            param.requires_grad = False
        for param in self.audio_net.parameters():
            param.requires_grad = False
            
        # 适配维度（关键修改）
        self.fusion = nn.Sequential(
            nn.Linear(8 + 8, 64),  # 假设两个模型各输出8维
            nn.ReLU(),
            nn.Linear(64, 8)
        )
    
    def forward(self, video_input, audio_input):
        # 视频输入形状: (B,16,3,224,224) → 模型输出 (B,8)
        video_out = self.video_net(video_input)
        
        # 音频输入形状: (B,216,40) → 模型输出 (B,8)
        audio_out = self.audio_net(audio_input)
        
        # 拼接特征
        fused = torch.cat([video_out, audio_out], dim=1)
        return self.fusion(fused)