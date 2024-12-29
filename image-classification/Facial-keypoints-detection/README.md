# Facial Keypoint Detection Project

This project demonstrates the process of detecting facial keypoints using a convolutional neural network (CNN) trained on the "Facial Key Points Detection" dataset from Kaggle.

## Demo Video

you can download the video and view it locally.

![Demo Video](https://github.com/MohamedAhmed35/computer-vision/blob/main/image-classification/Facial-keypoints-detection/bandicam%202024-12-29%2020-07-14-280.mp4)


## Model Download

You can download the trained model used for facial keypoint detection from the following link:

[Download the Trained Model](https://link_to_your_model_download)

## Project Overview

- **Dataset**: The model was trained using the "Facial Key Point Detection" dataset from Kaggle, which contains 5000 images of faces with corresponding facial landmarks.
- **Model**: A Convolutional Neural Network (CNN) was built to predict facial keypoints. The architecture consists of 5 convolutional layers with max-pooling and regularization (dropout = 0.3), followed by 3 dense layers.
- **Tools**: 
  - OpenCV for face detection and real-time video capture
  - Keras for model training and prediction
  - Haar Cascade for face detection
- **Input**: A video stream from the webcam.
- **Output**: The model predicts 68 landmarks, which are drawn as green circles on the detected face.

## Requirements

To run this project, you'll need to install the following Python packages:

- `opencv-python`
- `numpy`
- `keras`
- `tensorflow`
- `matplotlib`

You can install the required dependencies using `pip`:

```bash
pip install opencv-python numpy keras tensorflow matplotlib

