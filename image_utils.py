# image_utils.py

"""
Provides general image utilities for Modular-Stacker.
(Currently minimal, as many functions have been absorbed by other modules).
"""

import numpy as np
import cv2 # Still needed for image loading/saving in other modules, but not directly used here for common ops.

# This module is now intentionally minimal.
# Functions like apply_black_mask, normalize_image, and resize_image
# have been moved or their responsibilities absorbed by the LUT and XY pipeline.

# Example of a utility function that *might* be needed here in the future:
# def convert_to_grayscale_if_needed(image: np.ndarray) -> np.ndarray:
#     """Converts a color image to grayscale if it's not already."""
#     if len(image.shape) == 3 and image.shape[2] == 3: # Check for 3-channel BGR
#         return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     return image # Already grayscale or single channel

# No specific content needed for this module based on current plan,
# but keeping the file for organizational purposes.

if __name__ == '__main__':
    print("--- Image Utilities Module Test (Minimal) ---")
    print("This module is currently minimal, its core functionalities have been moved.")
    # Example of a dummy image for potential future tests
    dummy_image = np.zeros((50, 50), dtype=np.uint8)
    print(f"Dummy image created: shape={dummy_image.shape}, dtype={dummy_image.dtype}")
