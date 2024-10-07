import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the images
image1 = cv2.imread('C:/ML/LAB3/img2.png')
image2 = cv2.imread('C:/ML/LAB3/img4.png')

# Convert both images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Initialize SIFT detector
sift_detector = cv2.SIFT_create()

# Detect keypoints and compute descriptors for both images
keypoints1, descriptors1 = sift_detector.detectAndCompute(gray_image1, None)
keypoints2, descriptors2 = sift_detector.detectAndCompute(gray_image2, None)

# Check for empty descriptors
if descriptors1 is None or descriptors2 is None:
    print("Error: Could not compute descriptors for one or both images.")
else:
    # Display the number of descriptors for each image
    print(f"Number of keypoints in Image 1: {len(keypoints1)}")
    print(f"Number of descriptors in Image 1: {descriptors1.shape}")

    print(f"Number of keypoints in Image 2: {len(keypoints2)}")
    print(f"Number of descriptors in Image 2: {descriptors2.shape}")

    # Print the first descriptor to see what it looks like
    print(f"First descriptor in Image 1:\n{descriptors1[0]}")
    print(f"First descriptor in Image 2:\n{descriptors2[0]}")

    # Draw keypoints on both images (without matching yet)
    img1_with_keypoints = cv2.drawKeypoints(image1, keypoints1, None, color=(0, 255, 0), flags=0)
    img2_with_keypoints = cv2.drawKeypoints(image2, keypoints2, None, color=(0, 255, 0), flags=0)

    # Display the keypoints on both images
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img1_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Image 1')

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(img2_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.title('Keypoints in Image 2')

    plt.show()

    # Use BFMatcher to find matches between the descriptors
    matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)

    # Match descriptors using k-Nearest Neighbors (k=2)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

    # Filter matches using the ratio test
    filtered_matches = []
    for m, n in knn_matches:
        if m.distance < 0.75 * n.distance:
            filtered_matches.append(m)

    # Display the number of good matches found
    print(f"Number of good matches found: {len(filtered_matches)}")

    # Draw the matched keypoints on the images
    result_image = cv2.drawMatches(image1, keypoints1, image2, keypoints2, filtered_matches, None)

    # Display the matched image
    plt.figure(figsize=(10, 5))
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Feature Matching')
    plt.show()
