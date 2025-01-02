# R-CNN for Cat and Dog Detection using VOC 2007 Dataset  

## Project Overview  

This project implements a Region-based Convolutional Neural Network (R-CNN) for detecting cats and dogs in images using the VOC 2007 dataset. The model utilizes a pre-trained VGG16 architecture for feature extraction, SVM classifiers for categorization, and a regression model to adjust bounding box coordinates. The project addresses challenges such as class imbalance and employs techniques such as Non-Maximum Suppression to optimize detection outcomes.  

## Table of Contents  

- [Getting Started](#getting-started)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Usage](#usage)  
- [Model Architecture](#model-architecture)  
- [Results](#results)  
- [Contributing](#contributing)  
- [License](#license)  

## Getting Started  

Follow these instructions to set up the project for development and testing.  

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
