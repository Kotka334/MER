import os
import cv2
import glob
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger("VideoExtractor")

def extract_video_frames():
    """专门处理视频目录（Video_Song_Actor_01）中的MP4文件"""
    try:
        # ===================================================================
        # 1. 初始化人脸检测器
        # ===================================================================
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        if face_cascade.empty():
            logger.error("人脸检测器加载失败！")
            return

        # ===================================================================
        # 2. 递归扫描视频目录
        # ===================================================================
        mp4_pattern = os.path.join(
            Config.RAW_VIDEO_ROOT, 
            "**/actor_*/",  # 匹配所有actor目录
            "*.mp4"
        )
        video_files = glob.glob(mp4_pattern, recursive=True)
        
        if not video_files:
            logger.error(f"未找到视频文件！请检查路径：{Config.RAW_VIDEO_ROOT}")
            return
        logger.info(f"发现 {len(video_files)} 个视频文件")

        # ===================================================================
        # 3. 处理每个视频文件
        # ===================================================================
        for video_path in tqdm(video_files, desc="处理视频"):
            # 解析路径结构（示例：Video_Song_Actor_01/actor_01/xxx.mp4）
            path_parts = os.path.normpath(video_path).split(os.sep)
            actor_dir = next(p for p in path_parts if p.lower().startswith("actor_"))
            
            # 创建输出目录
            output_dir = os.path.join(
                Config.PROCESSED_VIDEO,
                "Video_Song_Actor_01",  # 添加数据集名称（需与实际数据集名一致）
                actor_dir,
                video_name
            )
            os.makedirs(output_dir, exist_ok=True)

            # ===============================================================
            # 4. 视频帧处理（原有逻辑保持不变）
            # ===============================================================
            cap = cv2.VideoCapture(video_path)
            saved_count = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 人脸检测与保存逻辑...
                
            cap.release()

        logger.info("视频处理完成！")

    except Exception as e:
        logger.error(f"处理失败：{str(e)}")
        raise

if __name__ == "__main__":
    extract_video_frames()