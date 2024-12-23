# Satellite Image Matching with LoFTR

## Overview
This project demonstrates a solution for matching satellite images using the LoFTR (Local Feature TRansformer) algorithm. The goal is to identify key features, establish correspondences between images.

## Solution Explanation

### Matching Algorithm
The LoFTR algorithm is used to detect key points and compute descriptors across images.

### RANSAC Filtering
To improve the robustness of the matches, RANSAC (Random Sample Consensus) is employed for outlier filtering. The algorithm helps in rejecting false matches by estimating the transformation model that best fits the inlier matches, improving the accuracy of keypoint correspondences.

### Downscaling
To efficiently handle large satellite images, the project implements a downscaling approach. The images are resized to a reduced resolution before matching, making the process more computationally efficient without losing crucial feature information. This is particularly useful for high-resolution satellite imagery.

### Data
The data used in this project comes from Sentinel-2 satellite imagery. The dataset includes images from different seasons to test the robustness of the matching algorithms under various conditions. The images are available for download from the provided dataset link.

## Project Structure

- `LoFTR_pipeline_setup.py`: Python script to set up the LoFTR pipeline for satellite image matching, including model configuration and setup.
- `loftr-demo.ipynb`: Jupyter Notebook that provides a detailed walkthrough of the solution with interactive code cells, including visualization of keypoints and matches.
- `requirements.txt`: File specifying the required Python packages and their versions for reproducibility.

## Setup

### Clone the Repository:
```bash
git clone <repository_url>
cd <repository_directory>
```

### Install Dependencies:
```bash
pip install -r requirements.txt
```

### Run the LoFTR Pipeline:
```bash
python LoFTR_pipeline_setup.py
```

### Explore the Jupyter Notebook:
Open `loftr-demo.ipynb` in a Jupyter environment for a step-by-step exploration of the solution, including interactive keypoint visualization and matching results.
