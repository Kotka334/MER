import os
import glob
from utils.config import Config

def find_dataset_dirs(pattern):
    """查找符合模式的数据集目录"""
    return glob.glob(os.path.join(Config.RAW_ROOT, pattern))

def get_actor_dirs(dataset_path):
    """获取数据集中的演员目录"""
    return [
        d for d in glob.glob(os.path.join(dataset_path, Config.ACTOR_PATTERN))
        if os.path.isdir(d)
    ]

def build_output_path(base_dir, dataset_name, actor_dir, file_basename):
    """构建输出路径"""
    return os.path.join(
        base_dir,
        dataset_name,
        os.path.basename(actor_dir),
        os.path.splitext(file_basename)[0]
    )