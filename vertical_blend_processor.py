"""
Core processing engine for vertical blending (receding and overhang gradients).
Migrated from the monolithic vertical blender and adapted for the Modular-Stacker pipeline.
This module processes a window of images and returns a window of processed images.
"""

import cv2
import numpy as np
from typing import List, Optional, Any

import lut_manager

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    lut_manager.set_config_reference(config_instance)

def _find_combined_mask(image_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Combines all white areas from a list of binary images into a single mask.
    """
    if not image_list:
        return None
    combined_mask = image_list[0].copy()
    for i in range(1, len(image_list)):
        combined_mask = cv2.bitwise_or(combined_mask, image_list[i])
    return combined_mask

def _calculate_gradient_field(
    current_mask: np.ndarray,
    comparison_mask: np.ndarray,
    fade_distance: float
) -> np.ndarray:
    """
    Calculates a normalized distance field radiating from the edges of the
    current_mask into the areas defined by the comparison_mask.
    """
    if comparison_mask is None or fade_distance <= 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    gradient_areas = cv2.bitwise_and(comparison_mask, cv2.bitwise_not(current_mask))
    if cv2.countNonZero(gradient_areas) == 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    distance_map = cv2.distanceTransform(cv2.bitwise_not(current_mask), cv2.DIST_L2, 5)
    gradient_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=gradient_areas)

    if np.max(gradient_distance_map) == 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    clipped_distance_map = np.clip(gradient_distance_map, 0, fade_distance)
    normalized_map = clipped_distance_map / fade_distance
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)
    
    return cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=gradient_areas)

def _merge_to_output(
    original_current_image: np.ndarray,
    receding_gradient: np.ndarray,
    overhang_gradient: np.ndarray
) -> np.ndarray:
    """
    Merges the calculated gradients onto the original current image.
    """
    output_image = np.zeros_like(original_current_image, dtype=np.uint8)
    
    if overhang_gradient is not None:
        overhang_mask = overhang_gradient > 0
        output_image[overhang_mask] = overhang_gradient[overhang_mask]

    if receding_gradient is not None:
        receding_mask = receding_gradient > 0
        output_image[receding_mask] = receding_gradient[receding_mask]

    current_shape_pixels = original_current_image > 0
    output_image[current_shape_pixels] = original_current_image[current_shape_pixels]

    return output_image

def process_image_window(image_window: List[np.ndarray]) -> List[np.ndarray]:
    """
    Main entry point. Processes an entire window of images, applying vertical blending
    to each image in the window relative to its neighbors.
    
    Args:
        image_window (List[np.ndarray]): A list of 8-bit grayscale images.
        
    Returns:
        List[np.ndarray]: A new list of processed 8-bit grayscale images.
    """
    if not image_window:
        raise ValueError("Image window cannot be empty.")
    if _config_ref is None:
        raise RuntimeError("Config reference not set in vertical_blend_processor.")

    processed_window = []
    num_images = len(image_window)
    binary_masks = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in image_window]

    for i in range(num_images):
        original_current_image = image_window[i]
        current_binary_mask = binary_masks[i]
        
        # --- Receding Gradient Calculation ---
        receding_gradient = None
        num_receding = _config_ref.vertical_receding_layers
        if num_receding > 0:
            prior_images = binary_masks[max(0, i - num_receding) : i]
            if prior_images:
                prior_mask = _find_combined_mask(prior_images)
                receding_gradient = _calculate_gradient_field(
                    current_binary_mask, prior_mask, _config_ref.vertical_receding_fade_dist
                )

        # --- Overhang Gradient Calculation ---
        overhang_gradient = None
        num_overhang = _config_ref.vertical_overhang_layers
        if num_overhang > 0:
            future_images = binary_masks[i + 1 : i + 1 + num_overhang]
            if future_images:
                future_mask = _find_combined_mask(future_images)
                # Invert masks to calculate distance from edges *inward*
                raw_overhang_gradient = _calculate_gradient_field(
                    cv2.bitwise_not(current_binary_mask),
                    current_binary_mask, # Gradient appears inside the current shape
                    _config_ref.vertical_overhang_fade_dist
                )
                # Mask the inward gradient by where future shapes will be
                overhang_gradient = cv2.bitwise_and(raw_overhang_gradient, raw_overhang_gradient, mask=future_mask)

        # --- Merge and Append ---
        processed_image = _merge_to_output(original_current_image, receding_gradient, overhang_gradient)
        processed_window.append(processed_image)
        
    return processed_window
