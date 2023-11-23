import cv2
import numpy as np
import time

# Function to process the images (resize, convert to grayscale, and apply cloud mask)
def process_image(img_path, cloud_mask_path=None, scale_factor=0.5):
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)
    
    # Apply cloud mask if provided
    if cloud_mask_path:
        cloud_mask = cv2.imread(cloud_mask_path, cv2.IMREAD_GRAYSCALE)
        cloud_mask = cv2.resize(cloud_mask, (img.shape[1], img.shape[0]))  # Resize to match the image size
        img = cv2.bitwise_and(img, img, mask=~cloud_mask)  # Apply the cloud mask
    
    # Check if the image is already in grayscale
    if len(img.shape) == 2:
        gray_img = img
    else:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
    return img, gray_img

# Function to match images using SIFT and FLANN Matcher
def match_images_sift_flann_with_mask(img1, img2, cloud_mask1=None, cloud_mask2=None, scale_factor=0.5, num_levels=3):
    sift = cv2.SIFT_create()
    bf = cv2.BFMatcher()

    # Downscale images to a specific level
    for _ in range(num_levels):
        img1 = cv2.pyrDown(img1)
        img2 = cv2.pyrDown(img2)

        if cloud_mask1 is not None:
            cloud_mask1 = cv2.resize(cloud_mask1, (img1.shape[1], img1.shape[0]))
            cloud_mask1 = cv2.pyrDown(cloud_mask1)
            
        if cloud_mask2 is not None:
            cloud_mask2 = cv2.resize(cloud_mask2, (img2.shape[1], img2.shape[0]))
            cloud_mask2 = cv2.pyrDown(cloud_mask2)

    # Apply cloud mask to the grayscale images
    if len(img1.shape) == 2:
        img1_gray = img1
    else:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    if cloud_mask1 is not None:
        img1_gray = cv2.bitwise_and(img1_gray, img1_gray, mask=~cloud_mask1)
        
    if len(img2.shape) == 2:
        img2_gray = img2
    else:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    if cloud_mask2 is not None:
        img2_gray = cv2.bitwise_and(img2_gray, img2_gray, mask=~cloud_mask2)

    # Detect keypoints and compute descriptors
    k1, d1 = sift.detectAndCompute(img1_gray, None)
    k2, d2 = sift.detectAndCompute(img2_gray, None)

    print(f'#keypoints in image1: {len(k1)}, image2: {len(k2)}')

    # Match the keypoints
    matches = bf.knnMatch(d1, d2, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    print(f'#good matches: {len(good_matches)} / {len(matches)}')

    # Calculate the ratio of good matches to total matches
    match_ratio = len(good_matches) / len(matches) if len(matches) > 0 else 0
    print(f'Match Ratio: {match_ratio:.2%}')

    # Visualize the matches
    img_matches = cv2.drawMatches(
        img1, k1, img2, k2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Display the visualization
    cv2.imshow("SIFT + FLANN Matcher with Cloud Mask", img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths to the input images (.jp2 format)
img1_path = '2017-01-01/T34JEP_20170101T082332_TCI.jp2'
img2_path = '2017-08-19/T34JEP_20170804T081559_TCI.jp2'

# Paths to the cloud masks (if available)
cloud_mask1_path = '2017-01-01/T34JEP_20170101T082332_B01.jp2'
cloud_mask2_path = '2017-08-19/T34JEP_20170804T081559_B01.jp2'

# Load and process the images with cloud masking
img1, gray_img1 = process_image(img1_path, cloud_mask1_path)
img2, gray_img2 = process_image(img2_path, cloud_mask2_path)

# Match the images using SIFT and FLANN Matcher with cloud masking at a specific level
match_images_sift_flann_with_mask(gray_img1, gray_img2, cloud_mask1=None, cloud_mask2=None, num_levels=3)





# Paths to the input images (.jp2 format)
#img1_path = '2017-01-01/T34JEP_20170101T082332_TCI.jp2'
#img2_path = '2017-08-19/T34JEP_20170804T081559_TCI.jp2'

# Paths to the cloud masks (if available)
#cloud_mask1_path = '2017-01-01/T34JEP_20170101T082332_B01.jp2'
#cloud_mask2_path = '2017-08-19/T34JEP_20170804T081559_B01.jp2'