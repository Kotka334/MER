import os

class Config:
    # 原始数据根目录（包含所有数据集）
    RAW_ROOT = "data/raw"
    
    # 处理后的数据路径
    PROCESSED_VIDEO = "data/processed/video_frames"
    PROCESSED_AUDIO = "data/processed/audio_features"
    
    # 数据识别模式
    AUDIO_DIR_PATTERN = "Audio_*"  # 匹配所有音频数据集目录
    VIDEO_DIR_PATTERN = "Video_*"  # 匹配所有视频数据集目录
    ACTOR_PATTERN = "actor_*"     # 匹配所有演员目录
    
    # DNN参数
    DNN_CONFIDENCE = 0.9
    DNN_INPUT_SIZE = (300, 300)
    # 图像尺寸（关键修复）
    IMG_SIZE = 224  # 人脸ROI的尺寸
    
    # 音频参数
    SAMPLE_RATE = 22050
    N_MFCC = 40
    AUDIO_EXTENSION = ".wav"

    # 训练参数
    BATCH_SIZE = 8
    VIDEO_EPOCHS = 15
    AUDIO_EPOCHS = 20
    FUSION_EPOCHS = 10
    LEARNING_RATE = {
        'video': 1e-4,
        'audio': 3e-4,
        'fusion': 5e-5
    }

# 自动创建目录
os.makedirs(Config.PROCESSED_VIDEO, exist_ok=True)
os.makedirs(Config.PROCESSED_AUDIO, exist_ok=True)