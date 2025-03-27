import os
import sys

# 将项目根目录加入Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import librosa
import numpy as np
from tqdm import tqdm
from utils.path_utils import (
    find_dataset_dirs,
    get_actor_dirs,
    build_output_path
)
from utils.config import Config
from utils.logger import setup_logger
import glob





logger = setup_logger("AudioProcessor")

class AudioProcessor:
    def process_all_datasets(self):
        """处理所有音频数据集"""
        audio_datasets = find_dataset_dirs(Config.AUDIO_DIR_PATTERN)
        if not audio_datasets:
            logger.error("未找到任何音频数据集目录！")
            return

        logger.info(f"发现 {len(audio_datasets)} 个音频数据集")
        
        for dataset_path in audio_datasets:
            dataset_name = os.path.basename(dataset_path)
            logger.info(f"正在处理数据集：{dataset_name}")
            
            actor_dirs = get_actor_dirs(dataset_path)
            if not actor_dirs:
                logger.warning(f"数据集 {dataset_name} 中没有演员目录")
                continue
                
            self.process_dataset(dataset_path, dataset_name, actor_dirs)

    def process_dataset(self, dataset_path, dataset_name, actor_dirs):
        """处理单个数据集"""
        for actor_dir in tqdm(actor_dirs, desc=f"处理 {dataset_name} 演员目录"):
            audio_files = glob.glob(os.path.join(actor_dir, f"*{Config.AUDIO_EXTENSION}"))
            if not audio_files:
                logger.debug(f"演员目录 {actor_dir} 中没有音频文件")
                continue
                
            for audio_path in audio_files:
                self.process_audio(audio_path, dataset_name)

    
    def process_audio(self, audio_path, dataset_name):
        """处理单个音频文件"""
        try:
        # ========== 关键修复1：统一路径格式 ==========
        # 规范化输入路径（解决原始路径中的混合分隔符问题）
            audio_path = os.path.normpath(audio_path)
        
        # ========== 关键修复2：统一actor目录命名 ==========
        # 获取actor目录名并强制转为小写（Windows大小写不敏感但Python严格匹配）
            actor_dir = os.path.basename(os.path.dirname(audio_path)).lower()  # 转为小写
        
        # ========== 关键修复3：绝对路径生成 ==========
        # 构建绝对输出路径（避免相对路径歧义）
            output_dir = os.path.normpath(
                os.path.join(
                    os.path.abspath(Config.PROCESSED_AUDIO),  # 转为绝对路径
                    dataset_name,
                    actor_dir  # 使用小写格式
                )
            )
        
        # ========== 关键修复4：增强路径存在性检查 ==========
            os.makedirs(output_dir, exist_ok=True)
            if not os.path.isdir(output_dir):
                raise NotADirectoryError(f"无法创建目录: {output_dir}")
        
        # 生成标准化输出文件名
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = os.path.normpath(
                os.path.join(output_dir, f"{base_name}.npy")
            )
        
        # ========== 关键修复5：调试日志 ==========
            logger.debug(f"输入音频路径: {audio_path}")
            logger.debug(f"生成输出路径: {output_path}")
            logger.debug(f"路径存在性检查: {os.path.exists(output_path)}")

        # 加载并处理音频（保持不变）
            y, sr = librosa.load(audio_path, sr=Config.SAMPLE_RATE)
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr,
                n_mfcc=Config.N_MFCC,
                n_fft=2048,
                hop_length=512
            )
            # 提取 MFCC 特征时固定 n_mfcc 参数
            mfcc = librosa.feature.mfcc(
            y=y, sr=sr,
            n_mfcc=40,  # 固定为 40 个系数
            n_fft=2048,
            hop_length=512
            )

        # ========== 关键修复6：保存前验证目录 ==========
            np.save(output_path, mfcc)
            logger.info(f"成功保存至: {output_path}")

        except Exception as e:
            logger.error(f"处理失败: {audio_path} → {str(e)}")
            raise  # 抛出异常以便追踪问题
if __name__ == "__main__":
    processor = AudioProcessor()
    processor.process_all_datasets()