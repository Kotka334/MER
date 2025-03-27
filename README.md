# MER-Emotion Recognition Project

## Overview

This project utilizes a multimodal emotion recognition model to fuse audio and video data for emotion recognition. The model is trained on **RAVDESS** dataset, consisting of speech and song recordings labeled with various emotional expressions. The primary goal is to classify emotions from both audio and video streams using deep learning techniques.

## Dataset Description

The dataset used for this project is the **RAVDESS** dataset, which contains emotional speech and song recordings. The dataset includes various emotions such as neutral, calm, happy, sad, angry, and fearful, spoken and sung by professional actors. 

You can download the **RAVDESS** dataset from the following link:

[Download RAVDESS Dataset](https://zenodo.org/records/1188976#.Xpaa3i-caAP)

## Project Setup

### 1. Clone the Repository

Start by cloning the repository:

```bash
git clone https://github.com/your-repository/emotion-recognition.git
```

### 2. Install Dependencies

After cloning the repository, navigate to the project directory and install the required dependencies:

```bash
cd MER
pip install -r requirements.txt
```

## How to Run

### Data Extraction

To extract the audio and video features, navigate to the `src/data_processing/` directory and run the following scripts:

#### Audio Extraction:
Run the audio extraction module:
```bash
python src/data_processing/audio_extractor.py
```
This will process the audio data from the **RAVDESS** dataset, extracting features such as **MFCC** which will be used for model training.

#### Video Extraction:
Run the video extraction module:
```bash
python src/data_processing/video_extractor_dnn.py
```
This will process the video data from the **RAVDESS** dataset, detecting faces in video frames using a pre-trained DNN model, and saving the frames for model training.

### Training the Model

Once the data extraction is complete, you can proceed to train the model by running the following command:
```bash
python src/training/train.py
```
This will start the model training, where both audio and video features will be passed to the **CNN + BiLSTM** model for emotion classification.

### Model Evaluation

After training the model, evaluate its performance using:
```bash
python src/training/evaluate.py
```
This will output performance metrics including accuracy, precision, recall, and F1 score for both validation and test sets.

## Acknowledgments

- **RAVDESS Dataset**: For providing the emotional speech and song recordings.
- **PyTorch**, **OpenCV**, **Librosa**, and other libraries used for deep learning and audio/video processing.

