from setuptools import setup, find_packages

setup(
    name="emotion_recognition",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.1',
        'librosa>=0.10.0',
        'opencv-python>=4.7.0',
        'dlib>=19.24.1',
        'pandas>=2.0.0'
    ],
    entry_points={
        'console_scripts': [
            'emotion-preprocess=emotion_recognition.preprocess:main',
            'emotion-train=emotion_recognition.train:main'
        ]
    }
)