# vertical_blend_processor.py

"""
Core processing engine for vertical blending (receding and overhang gradients).
This version has been updated to include legacy normalization and gamma controls,
and correctly applies LUTs to the individual gradient components before merging.
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
    _config_ref = config_instance
    lut_manager.set_config_reference(config_instance)

def _find_combined_mask(image_list: List[np.ndarray]) -> Optional[np.ndarray]:
    """Combines all white areas from a list of binary images into a single mask."""
    if not image_list: return None
    combined_mask = image_list[0].copy()
    for i in range(1, len(image_list)):
        combined_mask = cv2.bitwise_or(combined_mask, image_list[i])
    return combined_mask

def _calculate_gradient_field(
    current_mask: np.ndarray,
    comparison_mask: np.ndarray,
    fade_distance: float,
    use_fixed_fade: bool,
    gamma: float
) -> np.ndarray:
    """Calculates a normalized distance field as an 8-bit grayscale image."""
    if comparison_mask is None or (use_fixed_fade and fade_distance <= 0):
        return np.zeros_like(current_mask, dtype=np.uint8)

    gradient_areas = cv2.bitwise_and(comparison_mask, cv2.bitwise_not(current_mask))
    if cv2.countNonZero(gradient_areas) == 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    distance_map = cv2.distanceTransform(cv2.bitwise_not(current_mask), cv2.DIST_L2, 5)
    gradient_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=gradient_areas)

    if np.max(gradient_distance_map) == 0:
        return np.zeros_like(current_mask, dtype=np.uint8)

    if use_fixed_fade:
        normalized_map = np.clip(gradient_distance_map, 0, fade_distance) / fade_distance
    else:
        min_val, max_val, _, _ = cv2.minMaxLoc(gradient_distance_map, mask=gradient_areas)
        if max_val <= min_val: return np.zeros_like(current_mask, dtype=np.uint8)
        normalized_map = (gradient_distance_map - min_val) / (max_val - min_val)

    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * (inverted_normalized_map**gamma)).astype(np.uint8)
    
    return cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=gradient_areas)

def _merge_to_output(
    original_current_image: np.ndarray,
    receding_gradient: Optional[np.ndarray],
    overhang_gradient: Optional[np.ndarray]
) -> np.ndarray:
    """Merges gradients onto the original image, applying LUTs to gradients first."""
    output_image = np.zeros_like(original_current_image, dtype=np.uint8)
    
    # Apply LUTs to the individual 8-bit gradient maps before merging
    if overhang_gradient is not None:
        overhang_mask = overhang_gradient > 0
        if np.any(overhang_mask):
            overhang_luts_applied = lut_manager.apply_z_lut(overhang_gradient, "overhang")
            output_image[overhang_mask] = overhang_luts_applied[overhang_mask]

    if receding_gradient is not None:
        receding_mask = receding_gradient > 0
        if np.any(receding_mask):
            receding_luts_applied = lut_manager.apply_z_lut(receding_gradient, "receding")
            output_image[receding_mask] = receding_luts_applied[receding_mask]

    # Paste the original image's shape on top to preserve its anti-aliased edges
    current_shape_pixels = original_current_image > 0
    output_image[current_shape_pixels] = original_current_image[current_shape_pixels]

    return output_image

def process_image_window(image_window: List[np.ndarray]) -> List[np.ndarray]:
    """Processes a window of images, applying vertical blending to each."""
    if not image_window or _config_ref is None: return []

    processed_window = []
    num_images = len(image_window)
    original_images = [img.copy() for img in image_window]
    binary_masks = [cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1] for img in original_images]

    for i in range(num_images):
        receding_gradient = None
        if _config_ref.vertical_receding_layers > 0:
            prior_images = binary_masks[max(0, i - _config_ref.vertical_receding_layers) : i]
            if prior_images:
                prior_mask = _find_combined_mask(prior_images)
                receding_gradient = _calculate_gradient_field(
                    binary_masks[i], prior_mask, 
                    _config_ref.vertical_receding_fade_dist,
                    _config_ref.use_fixed_fade, _config_ref.vertical_gamma
                )

        overhang_gradient = None
        if _config_ref.vertical_overhang_layers > 0:
            future_images = binary_masks[i + 1 : i + 1 + _config_ref.vertical_overhang_layers]
            if future_images:
                future_mask = _find_combined_mask(future_images)
                raw_overhang_gradient = _calculate_gradient_field(
                    cv2.bitwise_not(binary_masks[i]), binary_masks[i],
                    _config_ref.vertical_overhang_fade_dist,
                    _config_ref.use_fixed_fade, _config_ref.vertical_gamma
                )
                overhang_gradient = cv2.bitwise_and(raw_overhang_gradient, raw_overhang_gradient, mask=future_mask)

        processed_image = _merge_to_output(original_images[i], receding_gradient, overhang_gradient)
        processed_window.append(processed_image)
        
    return processed_window

def blend_image_window_to_single_image(image_window: List[np.ndarray]) -> np.ndarray:
    """Blends a window of images into a single output image using vertical blending."""
    if not image_window: return np.zeros((100, 100), dtype=np.uint8)
    
    processed_window = process_image_window(image_window)
    if not processed_window: return np.zeros_like(image_window[0], dtype=np.uint8)

    stacked_images = np.stack(processed_window, axis=0)
    final_image = np.max(stacked_images, axis=0)
    return final_image.astype(np.uint8)
