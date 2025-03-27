import os
import glob
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger("DataLoader")

class MultimodalDataset(Dataset):
    def __init__(self, video_root, audio_root, max_time_steps=216, n_mfcc=40):
        self.max_time_steps = max_time_steps
        self.n_mfcc = n_mfcc  # 固定 MFCC 系数数量
        self.max_time_steps = max_time_steps  # 最大时间步长
        self.samples = []
        logger.info("初始化多模态数据集")

        # 遍历视频数据集目录（如 Video_Song_Actor_01）
        for video_dataset_dir in os.listdir(video_root):
            video_dataset_path = os.path.join(video_root, video_dataset_dir)
            if not os.path.isdir(video_dataset_path):
                continue

            # ================== 关键修复1：动态生成对应音频数据集目录名 ==================
            # 解析视频数据集类型（Song/Speech）
            if "Song" in video_dataset_dir:
                audio_dataset_dir = "Audio_Song_Actors_01-24"
            elif "Speech" in video_dataset_dir:
                audio_dataset_dir = "Audio_Speech_Actors_01-24"
            else:
                logger.error(f"未知数据集类型: {video_dataset_dir}")
                continue

            # 构建音频数据集路径
            audio_dataset_path = os.path.join(audio_root, audio_dataset_dir)
            if not os.path.exists(audio_dataset_path):
                logger.error(f"❌ 音频数据集不存在: {audio_dataset_path}")
                continue

            # ================== 关键修复2：统一演员目录命名（小写）==================
            # 遍历视频演员目录（如 actor_01）
            for video_actor_dir in os.listdir(video_dataset_path):
                # 生成对应的音频演员目录名（强制小写）
                actor_number = video_actor_dir.split('_')[-1]
                audio_actor_dir = f"actor_{actor_number}"  # 统一小写格式
                
                video_actor_path = os.path.join(video_dataset_path, video_actor_dir)
                audio_actor_path = os.path.join(audio_dataset_path, audio_actor_dir)
                
                if not os.path.exists(audio_actor_path):
                    logger.warning(f"演员目录不匹配: {audio_actor_path}")
                    continue

                # ================== 关键修复3：动态生成音频文件名 ==================
                # 遍历视频样本目录（如 01-02-01-01-01-01-01）
                for video_sample_dir in os.listdir(video_actor_path):
                    # 解析视频文件夹名（01-02-01-01-01-01-01）
                    parts = video_sample_dir.split('-')
                    if len(parts) != 7:
                        logger.warning(f"无效的视频文件夹名: {video_sample_dir}")
                        continue

                    # 动态生成对应的音频文件名（03-02-01-01-01-01-01.npy）
                    audio_sample_file = f"03-{parts[1]}-{parts[2]}-{parts[3]}-{parts[4]}-{parts[5]}-{parts[6]}.npy"
                    audio_sample_path = os.path.join(audio_actor_path, audio_sample_file)
                    
                    video_sample_path = os.path.join(video_actor_path, video_sample_dir)
                    
                    # 验证路径存在性
                    if not os.path.exists(audio_sample_path):
                        logger.warning(f"缺失音频特征: {audio_sample_path}")
                        continue

                    # 解析标签
                    try:
                        label = self._parse_label(video_sample_dir)
                    except Exception as e:
                        logger.error(f"标签解析失败: {video_sample_dir} → {str(e)}")
                        continue

                    self.samples.append({
                        'video': video_sample_path,
                        'audio': audio_sample_path,
                        'label': label
                    })
                    logger.debug(f"加载样本: {video_sample_dir} → 音频文件: {audio_sample_file}")

        logger.info(f"共加载 {len(self.samples)} 个有效样本")

    def _parse_label(self, filename):
        """RAVDESS文件名规范：
        03-01-06-01-02-01-24
        │  │  │  │  │  │  └─ Actor (24)
        │  │  │  │  │  └─ Repetition (01)
        │  │  │  │  └─ Statement ("02")
        │  │  │  └─ Intensity (01:normal, 02:strong)
        │  │  └─ Emotion (06:surprise)
        │  └─ Vocal channel (01:speech, 02:song)
        └─ Modality (01:audio-only, 02:video-only, 03:audio-video)
        """
        parts = filename.split('-')
        if len(parts) != 7:
            raise ValueError("Invalid filename format")
        
        emotion_code = int(parts[2])
        if not 1 <= emotion_code <= 8:
            raise ValueError(f"无效情绪编码: {emotion_code}")
        
        # 转换为0-based索引（neutral=0,...,surprised=7）
        return emotion_code - 1

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 加载视频帧
        video_frames = []
        frame_files = sorted(glob.glob(os.path.join(self.samples[idx]['video'], "face_*.jpg")))
        for frame_path in frame_files[:16]:  # 取前16帧
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (Config.IMG_SIZE, Config.IMG_SIZE))  # 确保统一尺寸
            frame = torch.FloatTensor(frame)  # 转换为 Tensor
            video_frames.append(frame)
        video_data = torch.stack(video_frames).permute(0, 3, 1, 2)  # (T, C, H, W)

        # 加载音频特征
        audio_data = np.load(self.samples[idx]['audio'])
        audio_data = self._pad_or_truncate_audio(audio_data)  # 统一时间步长
        audio_data = torch.FloatTensor(audio_data)  # 转换为 Tensor
        print(f"Audio shape: {audio_data.shape}, Video shape: {video_data.shape}")


        return {
            'video': video_data,
            'audio': audio_data,
            'label': torch.LongTensor([self.samples[idx]['label']])
        }

    def _pad_or_truncate_audio(self, audio_data):
        """统一音频特征的时序和 MFCC 系数维度"""
        T, F = audio_data.shape
        
        # 修正 MFCC 系数维度
        if F < self.n_mfcc:
            padded = np.zeros((T, self.n_mfcc))
            padded[:, :F] = audio_data
            audio_data = padded
        elif F > self.n_mfcc:
            audio_data = audio_data[:, :self.n_mfcc]
        
        # 修正时间步维度
        if T < self.max_time_steps:
            padded = np.zeros((self.max_time_steps, self.n_mfcc))
            padded[:T, :] = audio_data
            audio_data = padded
        elif T > self.max_time_steps:
            audio_data = audio_data[:self.max_time_steps, :]
        
        return audio_data


def create_dataloaders(batch_size=8, seed=42):
    dataset = MultimodalDataset(
        video_root="data/processed/video_frames",
        audio_root="data/processed/audio_features"
    )
    
    # 固定随机种子，保证结果可复现
    generator = torch.Generator().manual_seed(seed)

    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size  # 确保不丢样本
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=generator
    )
    
    return {
        'train': DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        'val': DataLoader(val_dataset, batch_size=batch_size),
        'test': DataLoader(test_dataset, batch_size=batch_size)
    }