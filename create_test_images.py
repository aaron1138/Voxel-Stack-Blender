import numpy as np
import cv2
import os

# Create a directory for the test images
output_dir = "tests/test_images"
os.makedirs(output_dir, exist_ok=True)

# Image dimensions
width, height = 100, 100

# --- Image 1 ---
img1 = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(img1, (25, 25), (75, 75), 255, -1)
cv2.imwrite(os.path.join(output_dir, "0001.png"), img1)

# --- Image 2 ---
img2 = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(img2, (25, 25), (75, 75), 255, -1)
cv2.imwrite(os.path.join(output_dir, "0002.png"), img2)

# --- Image 3 ---
img3 = np.zeros((height, width), dtype=np.uint8)
cv2.rectangle(img3, (35, 35), (65, 65), 255, -1)
cv2.imwrite(os.path.join(output_dir, "0003.png"), img3)

print("Test images created successfully.")
