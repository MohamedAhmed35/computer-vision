# Facial Keypoint Detection

This project is a facial keypoint detection system that predicts 68 facial landmarks (keypoints) from an image. The model was trained on a small version of the Flickr dataset, which contains images and corresponding landmark coordinates. The system detects facial landmarks in real-time using a webcam feed, processes the image to detect faces, and then predicts and displays the landmarks on the face.

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

