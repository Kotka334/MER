import os
import cv2
import librosa
import numpy as np
import torch
from utils.config import Config

def load_video_frames(video_dir, max_frames=16, img_size=224):
    """
    加载视频帧目录中的图像
    参数：
        video_dir: 包含视频帧的目录（如 face_0001.jpg, face_0002.jpg）
        max_frames: 最大帧数（与训练设置一致）
        img_size: 图像缩放尺寸
    返回：
        torch.Tensor: 形状为 (1, max_frames, 3, H, W) 的张量
    """
    frame_files = sorted(glob.glob(os.path.join(video_dir, "face_*.jpg")))
    frames = []
    
    for frame_path in frame_files[:max_frames]:
        # 读取并预处理帧
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (img_size, img_size))  # 调整尺寸
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # 转换通道顺序
        frame = frame.transpose(2, 0, 1)                 # (H, W, C) → (C, H, W)
        frames.append(frame)
    
    # 填充或截断
    if len(frames) < max_frames:
        padding = [np.zeros((3, img_size, img_size)) for _ in range(max_frames - len(frames))]
        frames += padding
    
    return torch.FloatTensor(np.array(frames)).unsqueeze(0)  # 添加batch维度

def extract_mfcc(audio_path, n_mfcc=40, max_length=216):
    """
    提取音频的MFCC特征
    参数：
        audio_path: 音频文件路径（支持.wav或.npy）
        n_mfcc: MFCC系数数量（与训练设置一致）
        max_length: 最大时间步长（与训练设置一致）
    返回：
        torch.Tensor: 形状为 (1, max_length, n_mfcc) 的张量
    """
    # 如果已经是预处理好的.npy文件
    if audio_path.endswith('.npy'):
        mfcc = np.load(audio_path)
    else:
        # 从.wav文件提取MFCC
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T  # (T, n_mfcc)
    
    # 填充或截断
    if mfcc.shape[0] > max_length:
        mfcc = mfcc[:max_length, :]
    elif mfcc.shape[0] < max_length:
        pad = np.zeros((max_length - mfcc.shape[0], n_mfcc))
        mfcc = np.concatenate([mfcc, pad], axis=0)
    
    return torch.FloatTensor(mfcc).unsqueeze(0)  # 添加batch维度