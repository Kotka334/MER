import torch
from torch import nn, optim
from models.fusion_net import SimpleFusion
from data_processing.dataloaders import create_dataloaders

def train_fusion():
    # 初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据（兼容原始数据格式）
    dataloaders = create_dataloaders(batch_size=8)
    
    # 加载预训练的单模态模型（需适配输入维度）
    video_model = VideoEmotionNet().eval().to(device)
    audio_model = AudioEmotionNet().eval().to(device)
    
    # 初始化融合模型
    fusion_model = SimpleFusion(video_model, audio_model).to(device)
    
    # 训练配置
    optimizer = optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # 训练循环（适配原始数据格式）
    for epoch in range(10):
        fusion_model.train()
        total_loss = 0.0
        
        for batch in dataloaders['train']:
            # 原始数据格式适配
            video = batch['video'].to(device)  # (B,16,3,224,224)
            audio = batch['audio'].to(device)  # (B,216,40)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = fusion_model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(dataloaders['train']):.4f}")
    
    # 保存模型
    torch.save(fusion_model.state_dict(), "fusion_model.pth")

if __name__ == "__main__":
    train_fusion()