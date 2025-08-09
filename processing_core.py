"""
Copyright (c) 2025 Aaron Baca

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

# processing_core.py (Modified)

import cv2
import numpy as np
import os

def load_image(filepath):
    """
    Loads an 8-bit grayscale image and creates a binary version (0 or 255).

    Args:
        filepath (str): The path to the image file.

    Returns:
        tuple: A tuple containing the binary image and the original grayscale image.
               Returns (None, None) if the image cannot be loaded.
    """
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Could not load image at {filepath}")
        return None, None
    # Ensure it's truly binary (0 for black, 255 for white) for the mask logic
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return binary_img, img

def find_prior_combined_white_mask(prior_images_list):
    """
    Combines all white areas from a list of prior binary images into a single mask.

    Args:
        prior_images_list (list): A list of numpy arrays representing prior binary images.

    Returns:
        numpy.ndarray: A single mask combining all white areas. Returns None if the list is empty.
    """
    if not prior_images_list:
        return None

    # Start with the first prior image's white areas
    combined_mask = prior_images_list[0].copy()

    # Logically OR with subsequent prior images
    for i in range(1, len(prior_images_list)):
        combined_mask = cv2.bitwise_or(combined_mask, prior_images_list[i])

    return combined_mask

def calculate_receding_gradient_field(current_white_mask, prior_white_combined_mask, use_fixed_normalization, fixed_fade_distance, debug_info=None):
    """
    Calculates a normalized distance field radiating from the edges of the current mask
    into areas that were white in prior layers.
    Removed 'gamma' parameter and its application from this function.

    Args:
        current_white_mask (numpy.ndarray): The binary mask of the current layer.
        prior_white_combined_mask (numpy.ndarray): The combined binary mask of prior layers.
        use_fixed_normalization (bool): Flag to use a fixed distance for normalization.
        fixed_fade_distance (float): The maximum distance for the fade gradient.
        debug_info (dict, optional): Information for saving intermediate debug images.
                                     Expected keys: 'output_folder', 'base_filename'.

    Returns:
        numpy.ndarray: The calculated gradient map as an 8-bit image.
    """
    if prior_white_combined_mask is None:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 1. Identify "receding white areas": Pixels that were white in prior layers but are now black.
    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))

    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03a_receding_white_areas.png"), receding_white_areas)

    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 2. Calculate the distance transform from the *current* shape's boundary.
    distance_transform_src = cv2.bitwise_not(current_white_mask)
    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03b_dist_src_for_transform.png"), distance_transform_src)
    
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    # 3. Mask the distance map to only include receding areas.
    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

    if np.max(receding_distance_map) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 4. Normalize the gradient.
    if use_fixed_normalization:
        clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
        # Avoid division by zero if fixed_fade_distance is 0
        denominator = fixed_fade_distance if fixed_fade_distance > 0 else 1.0
        normalized_map = (clipped_distance_map / denominator)
    else:
        min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=receding_white_areas)
        if max_val <= min_val:
            return np.zeros_like(current_white_mask, dtype=np.uint8)
        normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

    # 5. Invert and convert to 8-bit.
    # Removed gamma application here: (inverted_normalized_map ** gamma)
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

    # 6. Ensure the gradient only exists in the intended receding areas.
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)
    
    return final_gradient_map

def merge_to_output(original_current_image, receding_gradient):
    """
    Merges the calculated receding gradient with the original current image.
    The original image's pixels (especially anti-aliased edges) take precedence.

    Args:
        original_current_image (numpy.ndarray): The original, unmodified grayscale image for the current layer.
        receding_gradient (numpy.ndarray): The calculated gradient to blend in.

    Returns:
        numpy.ndarray: The final merged output image for the layer (8-bit).
    """
    # Use the lighter of the two values for each pixel.
    # This ensures that the original anti-aliasing is preserved and the gradient
    # smoothly blends from it.
    # The gradient exists where the original image is black, so max() works perfectly.
    output_image = np.maximum(original_current_image, receding_gradient)
    
    return output_image