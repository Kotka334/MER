import cv2
from utils.config import Config
from utils.logger import setup_logger

logger = setup_logger("VideoWriter")

def annotate_video(input_path, output_path):
    # 初始化视频流
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 初始化输出视频
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 加载人脸检测器（OpenCV）
    face_cascade = cv2.CascadeClassifier(Config.FACE_CASCADE_PATH)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # 转换为灰度图（加速检测）
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 人脸检测
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=Config.FACE_SCALE_FACTOR,
            minNeighbors=Config.FACE_MIN_NEIGHBORS
        )
        
        # 绘制检测框（情绪预测需后续添加）
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        out.write(frame)
    
    cap.release()
    out.release()
    logger.info(f"视频已保存至 {output_path}")