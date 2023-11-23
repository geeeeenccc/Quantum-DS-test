# Satellite Image Matching with SIFT and BFMatcher

## Overview

This project demonstrates a solution for matching satellite images using the Scale-Invariant Feature Transform (SIFT) algorithm and the Brute-Force Matcher from the OpenCV library. The aim is to identify key features and establish correspondences between images, with a focus on handling large satellite images and incorporating cloud masking for better accuracy.

## Solution Explanation

### Matching Algorithm

The SIFT algorithm is used to detect key points and compute descriptors for each image. The Brute-Force Matcher is then used to find matches between the descriptors of the two images. Ratio testing is applied to filter out unrealistic matches, and the remaining matches are visualized.

### Cloud Masking

Cloud masking is incorporated to improve the accuracy of keypoint detection and matching. The project provides an option to input cloud masks, allowing the algorithm to exclude keypoints in regions covered by clouds during the matching process.

### Downscaling

To handle large satellite images, a downscaling approach is implemented. The images are iteratively downsampled to a specified number of levels, making the matching process more efficient without losing key features.

## Project Structure

- **SIFT_BFMatcher.py**: Python script containing the SIFT and BFMatcher implementation with cloud masking.

- **sift-BFMatcher.ipynb**: Jupyter Notebook providing a detailed walkthrough of the solution with interactive code cells.

- **requirements.txt**: File specifying the required Python packages and their versions for reproducibility.

## Setup

1. **Clone the Repository:**

   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```

2. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the SIFT Matching Script:**

   ```bash
   python SIFT_BFMatcher.py
   ```

   Follow the prompts and input the paths to your satellite images and cloud masks if available.

4. **Explore the Jupyter Notebook:**

   Open `sift-BFMatcher.ipynb` in a Jupyter environment for a step-by-step exploration of the solution.

## Notes

- Ensure that your satellite images are in the .jp2 format.
- Cloud masks are optional but recommended for better accuracy.
- Adjust parameters in the script to suit your specific use case.
