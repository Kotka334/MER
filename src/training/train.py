import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入 MultimodalDataset
from src.data_processing.dataloaders import MultimodalDataset

# 其他导入
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.audio_net import AudioEmotionNet
from models.video_net import VideoEmotionNet
from models.fusion_net import SimpleFusion
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score


class Trainer:
    def __init__(self, model, dataloaders, modality='audio', device='cuda'):
        """
        初始化训练器
        :param model: 模型实例
        :param dataloaders: 包含 'train' 和 'val' 的 DataLoader 字典
        :param modality: 模态类型 ('audio', 'video', 'fusion')
        :param device: 训练设备 ('cuda' 或 'cpu')
        """
        self.model = model.to(device)
        self.dataloaders = dataloaders
        self.modality = modality
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        total_loss = 0.0
        for batch in tqdm(self.dataloaders['train'], desc='Training'):
            inputs = batch[self.modality].to(self.device)
            labels = batch['label'].squeeze().to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloaders['train'])

    def test(self):
        """在测试集上评估模型性能"""
        self.model.eval()
        correct = 0
        all_preds = []
        all_labels = []
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.dataloaders['test'], desc='Testing'):
                inputs = batch[self.modality].to(self.device) if self.modality != 'both' else \
                         torch.cat((batch['audio'], batch['video']), dim=1).to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                correct += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        f1 = f1_score(all_labels, all_preds, average='weighted')
        acc = correct / len(self.dataloaders['test'].dataset)
        avg_loss = total_loss / len(self.dataloaders['test'])
        print(f"[Test] Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Loss: {avg_loss:.4f}")
        return acc, f1, avg_loss


    def validate(self):
        """验证模型并计算 F1 Score"""
        self.model.eval()
        correct = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in tqdm(self.dataloaders['val'], desc='Validating'):
                inputs = batch[self.modality].to(self.device)
                labels = batch['label'].squeeze().to(self.device)
                outputs = self.model(inputs)
                preds = outputs.argmax(dim=1)
                
                # 计算准确率
                correct += (preds == labels).sum().item()

                # 保存所有预测和标签，以便计算 F1 Score
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算 F1 Score
        f1 = f1_score(all_labels, all_preds, average='weighted')  # 使用加权平均 F1 Score
        acc = correct / len(self.dataloaders['val'].dataset)

        print(f"Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
        
        return acc, f1  # 返回准确率和 F1 Score

    def train(self, epochs=10, save_path=None):
        history = {
            'train_loss': [],
            'val_acc': [], 'val_f1': [],
            'test_acc': [], 'test_f1': [], 'test_loss': []
        }
        best_acc = 0.0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_acc, val_f1 = self.validate()
            test_acc, test_f1, test_loss = self.test()

            history['train_loss'].append(train_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)
            history['test_acc'].append(test_acc)
            history['test_f1'].append(test_f1)
            history['test_loss'].append(test_loss)

            print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Val Acc={val_acc:.4f}, Test Acc={test_acc:.4f}")

            if val_acc > best_acc:
                best_acc = val_acc
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"模型已保存至 {save_path}")

        return history




def plot_history(history, title):
    """绘制训练过程曲线，包括验证和测试准确率"""
    plt.figure(figsize=(12, 5))

    # Loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title(f"{title} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # Accuracy 曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title(f"{title} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()



def collate_fn(batch):
    # 调试：检查每个样本的 video 和 audio 类型
    for item in batch:
        if not isinstance(item['video'], torch.Tensor):
            print("Invalid video data type:", type(item['video']))
        if not isinstance(item['audio'], torch.Tensor):
            print("Invalid audio data type:", type(item['audio']))
    
    video = torch.stack([item['video'] for item in batch])
    audio = torch.stack([item['audio'] for item in batch])
    label = torch.cat([item['label'] for item in batch])
    return {'video': video, 'audio': audio, 'label': label}


def create_dataloaders(batch_size=8, seed=42):
    dataset = MultimodalDataset(
        video_root='data/processed/video_frames',
        audio_root='data/processed/audio_features'
    )
    
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=generator
    )
    
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn),
        'val': DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn),
        'test': DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)
    }


# 训练流程
if __name__ == "__main__":
    # 创建 DataLoader
    dataloaders = create_dataloaders(batch_size=8)

    # 训练音频模型
    audio_model = AudioEmotionNet()
    audio_trainer = Trainer(audio_model, dataloaders, modality='audio', device='cuda')
    audio_history = audio_trainer.train(epochs=10)
    plot_history(audio_history, "Audio Model")

    # 训练视频模型
    video_model = VideoEmotionNet()
    video_trainer = Trainer(video_model, dataloaders, modality='video', device='cuda')
    video_history = video_trainer.train(epochs=10)
    plot_history(video_history, "Video Model")

    # 训练融合模型
    fusion_net = MultimodalFusionNet(audio_model, video_model)
    fusion_trainer = Trainer(fusion_net, dataloaders, modality='both', device='cuda')
    fusion_history = fusion_trainer.train(epochs=5)
    plot_history(fusion_history, "Fusion Model")