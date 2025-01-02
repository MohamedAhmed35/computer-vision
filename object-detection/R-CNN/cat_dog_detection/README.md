# Cat and Dog Detection using R-CNN  

This project demonstrates the process of detecting cats and dogs in images using a Region-based Convolutional Neural Network (R-CNN) trained on the VOC 2007 dataset.  

## Model Download  

You can download the trained model used for cat and dog detection from the following link:  

[Download the Trained Model](https://drive.google.com/uc?id=YOUR_MODEL_FILE_ID)  

> **Note:** Replace `YOUR_MODEL_FILE_ID` with the actual file ID from your Google Drive link.  

## Project Overview  

- **Dataset**: The model was trained using the VOC 2007 dataset, which contains thousands of images of animals with corresponding bounding box annotations. Both the original and processed images are available for download from Google Drive:  
  - [Download Original and Processed Images](https://drive.google.com/uc?id=YOUR_DATASET_FILE_ID)  

- **Model**: A Region-based Convolutional Neural Network (R-CNN) architecture was employed, utilizing a pre-trained VGG16 model for feature extraction. The model consists of:  
  - Feature extraction using VGG16  
  - Support Vector Machines (SVM) for classification  
  - A regression model for bounding box adjustments  

- **Tools**:  
  - OpenCV for image processing and visualization  
  - Keras for model training and prediction  
  - Selective Search for region proposal generation  
  - Non-Maximum Suppression (NMS) for refining bounding boxes  

- **Input**: A collection of images from the VOC 2007 dataset.  

- **Output**: The model predicts bounding boxes and class labels for detected cats and dogs in the images.  

## Challenges Faced  

During the development of this project, several challenges were encountered:  

1. **Class Imbalance**: The dataset contained an uneven distribution of images for cats and dogs, which required the implementation of techniques to balance the training process.  

2. **Complex Object Detection**: Accurately detecting objects in varying contexts and orientations proved challenging, necessitating careful tuning of the region proposal algorithm.  

3. **Bounding Box Precision**: Ensuring accurate bounding box predictions involved developing a regression model to adjust the bounding boxes effectively.  

4. **Non-Maximum Suppression (NMS)**: Implementing NMS to filter overlapping bounding boxes required careful consideration of the IoU calculations.  

## Requirements  

To run this project, you'll need to install the following Python packages:  

- opencv-python  
- numpy  
- keras  
- tensorflow  
- matplotlib  
- scikit-learn  

You can install the required dependencies using pip:  

```bash  
pip install opencv-python numpy keras tensorflow matplotlib scikit-learn
