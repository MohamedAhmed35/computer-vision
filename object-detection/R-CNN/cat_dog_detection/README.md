# RCNN Pipeline for Cats and Dogs Detection

## Overview
THis project implements an R-CNN **pipeline** inspired by the groudbreaking **R-CNN paper (2014)** by Ross Grishick et al. The goal is to detect and localize cats and dogs in images from the **VOC 2007 dataset** using a manual designed pipeline.
While modern models like YOLO and Faster R-CNN provide faster and more accurate solutions, this project focuses on understanding and recreating the original R-CNN methodology, which served as the foundation for these advancements.

## Key Features
1. Region Proposal Generation
   - Generated candidate regions using Selective Search, limiting proposals to 50 per image for computational efficiency.
