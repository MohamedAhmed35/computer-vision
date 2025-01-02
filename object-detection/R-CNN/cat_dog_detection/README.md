# Cat and Dog Detection using R-CNN  

This project demonstrates the process of detecting cats and dogs in images using a Region-based Convolutional Neural Network (R-CNN) trained specifically on selected images of cats and dogs from the VOC 2007 dataset.  

## Model Download  

You can download the trained models used for cat and dog detection from the following links:  

- [Download the Trained Model for Cats](https://drive.google.com/uc?id=YOUR_CATS_MODEL_FILE_ID)  
- [Download the Trained Model for Dogs](https://drive.google.com/uc?id=YOUR_DOGS_MODEL_FILE_ID)  

> **Note:** Replace `YOUR_CATS_MODEL_FILE_ID` and `YOUR_DOGS_MODEL_FILE_ID` with the actual file IDs from your Google Drive links.  

## Project Overview  

- **Dataset**: The model was trained using cat and dog images from the VOC 2007 dataset, which includes annotated images of various animals. The original images and their annotations were utilized to extract only the relevant classes (cats and dogs).   

- **Region Proposals**: For each selected image, the Selective Search algorithm was applied to generate 50 region proposals. These proposals help in identifying potential bounding boxes for cats and dogs in the images. The generated region proposal images, referred to as "processed images," were saved during this step for training and evaluation.  

- **Model**: A Region-based Convolutional Neural Network (R-CNN) architecture was employed, utilizing a pre-trained VGG16 model for feature extraction. The model consists of:  
  - Feature extraction using VGG16  
  - Support Vector Machines (SVM) for classification of the features  
  - A regression model to adjust the bounding box predictions  

- **Tools**:  
  - OpenCV for image processing and visualization  
  - Keras for model training and prediction  
  - Selective Search for region proposal generation  
  - Non-Maximum Suppression (NMS) for refining bounding boxes  

- **Input**: A collection of processed images generated through the Selective Search method, showing potential regions where cats and dogs may be located.  

- **Output**: The model predicts bounding boxes and class labels for detected cats and dogs in the processed images.  

## Challenges Faced  

During the development of this project, several challenges were encountered:  

1. **Class Imbalance**: The dataset contained a higher number of images of one class compared to the other. This imbalance required careful handling during training.  

2. **Complex Object Detection**: Accurately detecting animals within diverse contexts and orientations proved challenging, necessitating the fine-tuning of the region proposal algorithm.  

3. **Bounding Box Precision**: Ensuring accurate bounding boxes involved developing a regression model that could effectively adjust the bounding boxes based on predicted coordinates.  

4. **Non-Maximum Suppression (NMS)**: Implementing NMS to filter out overlapping bounding boxes required careful IoU calculations to retain valid detections while eliminating redundancy.  

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
