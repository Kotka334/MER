import os
import sys

# 将项目根目录加入Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
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

logger = setup_logger("VideoProcessor")

class VideoProcessor:
    def __init__(self):
        self.net = cv2.dnn.readNetFromCaffe(
            "models/deploy.prototxt",
            "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
        )
    
    def process_all_datasets(self):
        """处理所有视频数据集"""
        video_datasets = find_dataset_dirs(Config.VIDEO_DIR_PATTERN)
        if not video_datasets:
            logger.error("未找到任何视频数据集目录！")
            return

        logger.info(f"发现 {len(video_datasets)} 个视频数据集")
        
        for dataset_path in video_datasets:
            dataset_name = os.path.basename(dataset_path)
            logger.info(f"正在处理数据集：{dataset_name}")
            
            # 修正1: 使用正确变量名actor_dirs
            actor_dirs = get_actor_dirs(dataset_path)
            if not actor_dirs:
                logger.warning(f"数据集 {dataset_name} 中没有演员目录")
                continue
                
            self.process_dataset(dataset_path, dataset_name, actor_dirs)

    def process_dataset(self, dataset_path, dataset_name, actor_dirs):
        """处理单个数据集"""
        for actor_dir in tqdm(actor_dirs, desc=f"处理 {dataset_name} 演员目录"):
            # 修正2: 正确构建视频文件路径
            video_files = glob.glob(os.path.join(actor_dir, "*.mp4"))
            if not video_files:
                logger.debug(f"演员目录 {actor_dir} 中没有MP4文件")
                continue
                
            for video_path in video_files:
                self.process_video(video_path, dataset_name, actor_dir)

    # 修正3: 添加缺失参数actor_dir
    def process_video(self, video_path, dataset_name, actor_dir):
        """处理单个视频文件"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"无法打开视频文件：{video_path}")
                return

            # 修正4: 正确生成video_name
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            
            # 修正5: 完整路径生成逻辑
            output_dir = os.path.join(
                Config.PROCESSED_VIDEO,
                dataset_name,  # 使用数据集名称
                os.path.basename(actor_dir),  # 演员目录名
                video_name
            )
            os.makedirs(output_dir, exist_ok=True)

            saved_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                faces = self.detect_faces(frame)
                if len(faces) == 1:
                    self.save_face(frame, faces[0], output_dir, saved_count)
                    saved_count += 1

            logger.debug(f"保存 {saved_count} 帧到 {output_dir}")

        except Exception as e:
            logger.error(f"处理视频出错：{video_path} → {str(e)}")
        finally:
            cap.release()

    def detect_faces(self, frame):
        """DNN人脸检测"""
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, Config.DNN_INPUT_SIZE), 
            1.0, 
            Config.DNN_INPUT_SIZE,
            (104.0, 177.0, 123.0)
        )
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > Config.DNN_CONFIDENCE:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                faces.append(box.astype("int"))
        return faces

    def save_face(self, frame, box, output_dir, index):
        """保存人脸ROI"""
        x1, y1, x2, y2 = box
        face_roi = frame[y1:y2, x1:x2]
        output_path = os.path.join(output_dir, f"face_{index:04d}.jpg")
        cv2.imwrite(output_path, face_roi)

if __name__ == "__main__":
    processor = VideoProcessor()
    processor.process_all_datasets()