# utils/logger.py
import logging
import os
from datetime import datetime

def setup_logger(name):
    """
    创建一个配置好的日志记录器
    Args:
        name (str): 日志名称（通常用__name__）
    Returns:
        logging.Logger: 配置好的日志对象
    """
    # 创建日志对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（确保logs目录存在）
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f"{datetime.now().strftime('%Y%m%d')}.log")
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger