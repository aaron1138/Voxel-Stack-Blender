import numpy as np
import cv2
import os

def create_test_images(output_dir="test_images", num_images=10, width=100, height=100):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(num_images):
        img = np.zeros((height, width), dtype=np.uint8)

        # Add a white rectangle that moves across the image
        rect_width = 20
        rect_height = 80
        start_x = int((i / num_images) * (width - rect_width))
        start_y = (height - rect_height) // 2

        cv2.rectangle(img, (start_x, start_y), (start_x + rect_width, start_y + rect_height), 255, -1)

        cv2.imwrite(os.path.join(output_dir, f"slice_{i:04d}.png"), img)

if __name__ == "__main__":
    create_test_images()
