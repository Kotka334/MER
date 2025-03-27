import torch
from torch.utils.data import DataLoader
from fusion_net import WeightedFusion
from dataloaders import create_dataloaders

# 训练配置
EPOCHS = 10
BATCH_SIZE = 8
SAVE_PATH = "models/fusion_model.pth"

def main():
    # 设备检测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据
    dataloaders = create_dataloaders(batch_size=BATCH_SIZE)
    
    # 加载预训练模型
    video_model = VideoEmotionNet().eval().to(device)
    video_model.load_state_dict(torch.load('models/video_model.pth', map_location=device))
    
    audio_model = AudioEmotionNet().eval().to(device)
    audio_model.load_state_dict(torch.load('models/audio_model.pth', map_location=device))
    
    # 初始化融合模型
    fusion_model = WeightedFusion(video_model, audio_model).to(device)
    
    # 训练设置
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    
    # 训练循环
    for epoch in range(EPOCHS):
        fusion_model.train()
        total_loss = 0.0
        
        for batch in dataloaders['train']:
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].squeeze().to(device)
            
            optimizer.zero_grad()
            outputs = fusion_model(video, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloaders['train'])
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(fusion_model.state_dict(), SAVE_PATH)
    print(f"融合模型已保存至 {SAVE_PATH}")

if __name__ == "__main__":
    main()