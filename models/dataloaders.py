import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MultimodalDataset(Dataset):
    def __init__(self, video_root, audio_root, max_time_steps=16, n_mfcc=40):
        self.max_time_steps = max_time_steps
        self.n_mfcc = n_mfcc
        self.samples = []
        
        # 遍历数据集目录（保持原有逻辑）
        for video_dir in os.listdir(video_root):
            video_path = os.path.join(video_root, video_dir)
            audio_path = os.path.join(audio_root, video_dir.replace("Video", "Audio"))
            
            if not os.path.exists(audio_path):
                continue
                
            # 遍历样本目录
            for actor_dir in os.listdir(video_path):
                actor_video = os.path.join(video_path, actor_dir)
                actor_audio = os.path.join(audio_path, actor_dir)
                
                for sample_dir in os.listdir(actor_video):
                    video_sample = os.path.join(actor_video, sample_dir)
                    audio_sample = os.path.join(actor_audio, sample_dir + ".npy")
                    
                    if os.path.exists(audio_sample):
                        self.samples.append({
                            'video': video_sample,
                            'audio': audio_sample,
                            'label': int(sample_dir.split('-')[2]) - 1
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 加载视频帧（完整处理）
        video_frames = []
        frame_files = sorted(glob.glob(os.path.join(self.samples[idx]['video'], "*.jpg"))[:16])
        
        for frame_path in frame_files:
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (224, 224))  # 调整尺寸
            frame = frame.transpose(2, 0, 1)       # 转为 (C,H,W)
            video_frames.append(torch.FloatTensor(frame))
        
        # 处理视频维度 (T,C,H,W)
        video_data = torch.stack(video_frames)
        if len(video_frames) < 16:
            padding = torch.zeros((16 - len(video_frames), 3, 224, 224))
            video_data = torch.cat([video_data, padding], dim=0)

        # 加载并处理音频
        audio_data = np.load(self.samples[idx]['audio'])
        if audio_data.shape[0] > 16:
            audio_data = audio_data[:16, :]
        elif audio_data.shape[0] < 16:
            pad = np.zeros((16 - audio_data.shape[0], 40))
            audio_data = np.concatenate([audio_data, pad], axis=0)

        return {
            'video': video_data,          # (16,3,224,224)
            'audio': torch.FloatTensor(audio_data),  # (16,40)
            'label': torch.LongTensor([self.samples[idx]['label']])
        }

def create_dataloaders(batch_size=8):
    dataset = MultimodalDataset(
        video_root="data/processed/video_frames",
        audio_root="data/processed/audio_features"
    )
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    return {
        'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_set, batch_size=batch_size)
    }