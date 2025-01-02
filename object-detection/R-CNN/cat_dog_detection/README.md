# R-CNN for Cat and Dog Detection using VOC 2007 Dataset  

## Project Overview  

This project implements a Region-based Convolutional Neural Network (R-CNN) for detecting cats and dogs in images using the VOC 2007 dataset. The goal is to accurately identify and localize cats and dogs within various images leveraging deep learning techniques. The model incorporates a pre-trained VGG16 architecture for feature extraction and employs Support Vector Machines (SVMs) for classification.  

## Data  

The VOC 2007 dataset is quite large and includes annotated images, which makes it unsuitable for direct uploads to GitHub. As such, the original dataset and the processed images are stored on Google Drive. You can download the dataset from the following link:  

[Download VOC 2007 Dataset](https://drive.google.com/uc?id=YOUR_FILE_ID)  

> **Note:** Please replace `YOUR_FILE_ID` with the actual file ID from your Google Drive link.  

## Project Structure  

The GitHub repository contains the following:  

- **Notebooks**: Jupyter notebooks containing the model training and evaluation code.  
- **Scripts**: Python scripts for data processing, model training, and evaluation functions.  
- **requirements.txt**: A list of dependencies required to run the project.  

## Challenges Faced  

During the development of this project, several challenges emerged:  

1. **Data Imbalance**: The dataset contained an uneven distribution of images between cats and dogs, leading to biased classification results. Techniques such as `class_weight='balanced'` were implemented in SVM training to mitigate this.  

2. **Complexity of Object Detection**: Accurately detecting objects and adjusting bounding boxes based on predicted features proved challenging. A custom regression model was developed to adjust bounding box coordinates effectively.  

3. **Region Proposal Generation**: Implementing the Selective Search algorithm required careful tuning to balance the number of proposals and processing time, ensuring relevant regions were highlighted for detection.  

4. **Non-Maximum Suppression (NMS)**: Implementing NMS to filter out redundant bounding boxes was complex, especially due to challenges in obtaining reliable ground truth bounding box coordinates for IoU calculations. This was addressed by adapting methods inspired by popular frameworks like YOLO.  

## Usage  

### Prerequisites  

Before you begin, ensure you have the following installed:  

- Python 3.x  
- pip (Python package manager)  
- git  

### Installation  

1. Clone the repository to your local machine:  

   ```bash  
   git clone https://github.com/yourusername/cat-dog-detection-rcnn.git  
   cd cat-dog-detection-rcnn
