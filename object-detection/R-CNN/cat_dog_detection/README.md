# RCNN Pipeline for Cats and Dogs Detection

## Overview
THis project implements an R-CNN **pipeline** inspired by the groudbreaking **R-CNN paper (2014)** by Ross Grishick et al. The goal is to detect and localize cats and dogs in images from the **VOC 2007 dataset** using a manual designed pipeline.
While modern models like YOLO and Faster R-CNN provide faster and more accurate solutions, this project focuses on understanding and recreating the original R-CNN methodology, which served as the foundation for these advancements.

## Key Features
### 1. Region Proposal Generation
   - Generated candidate regions using **Selective Search**, limiting proposals to 50 per image for computational efficiency.
   - Annotated regions with class labels and bounding box coordinates, saved in structured CSV files.
### 2. Feature Extraction
   - Leveraged a pre-trained **VGG16 network** (excluding dense layers) to extract 4096-dimensional feature vectors for each proposal.
   - These vectors served as inputs for both classification and bounding box refinement tasks.
### 3. Classification with SVMs
   - Trained **SVM classifiers** to differentiate between cats, dogs, and background
### 4. Bounding Box Refinement
   - Built a **bounding box regressor** using VGG16's deeper layers to predict transformation parameters for refining object localization.
   - Calculated transformation parameters based on the original ground truth annotations.
   - Because the transformation parameter was calculated based on IOU between 0.6 and 1 (as the paper suggested), the regressor was not giving accurate results when used to adjust the predicted bounding box.
### 5. Testing and Optimization
   - **Softmax-based scoring** to prioritize predictions by confidence.
   - **Non-Maximum Suppression (NMS)** adapted from YOLO, using dynamically selected bounding boxes as ground truth for IoU calculations, to remove redundant bounding boxes with overlapping regions.

## Results
The pipeline successfully detects and localizes cats and dogs, though it highlights the computational bottlenecks of the original R-CNN approach. This project serves as an educational tool for understanding the evolution of object detection.
   - Bounding box Refinement
     ![Example Image](/images/results/adjust_bounding_box_1.png)  


## Repository Structure
- **Notebooks**: All code is provided in Jupter notebook
- **helperfunction.py**: A python file contains the function that used across the notebooks
- **External Resources**:
     - [data](https://drive.google.com/drive/folders/15jX5IyXv8K4tAyII1IQaKfFVtYE63Wbi?usp=drive_link)
     - [trained models](https://drive.google.com/drive/folders/13xcM_Hp2dmNIq9hgr6GMmlFRXNEIk3iS?usp=drive_link)
