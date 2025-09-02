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
from scipy import ndimage
import numba
import tiledb

from config import ProcessingMode

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

def identify_rois(binary_image, min_size=100):
    """
    Identifies connected components (ROIs) in a binary image that meet a minimum size criteria.

    Args:
        binary_image (numpy.ndarray): The input binary image (0 or 255).
        min_size (int): The minimum number of pixels for a component to be considered an ROI.

    Returns:
        list: A list of dictionaries, where each dictionary represents an ROI and contains:
              'label' (int), 'area' (int), 'bbox' (tuple), 'centroid' (tuple), 'mask' (ndarray).
              Returns an empty list if no components are found.
    """
    rois = []
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image, 8, cv2.CV_32S)

    # Start from label 1, as label 0 is the background
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_size:
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]

            # Create a mask for the current ROI
            roi_mask = (labels == i).astype(np.uint8) * 255

            rois.append({
                'label': i,
                'area': area,
                'bbox': (x, y, w, h),
                'centroid': centroids[i],
                'mask': roi_mask
            })

    return rois

def _calculate_receding_gradient_field_fixed_fade(current_white_mask, prior_white_combined_mask, config, debug_info=None):
    """
    Calculates a normalized distance field radiating from the edges of the current mask
    into areas that were white in prior layers. (Original Fixed Fade implementation)
    """
    use_fixed_normalization = config.use_fixed_fade_receding
    fixed_fade_distance = config.fixed_fade_distance_receding

    if prior_white_combined_mask is None:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 1. Identify "receding white areas"
    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03a_receding_white_areas.png"), receding_white_areas)
    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 2. Calculate distance transform
    distance_transform_src = cv2.bitwise_not(current_white_mask)
    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03b_dist_src_for_transform.png"), distance_transform_src)
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    # 3. Mask distance map
    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)
    if np.max(receding_distance_map) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 4. Normalize gradient
    if use_fixed_normalization:
        clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
        denominator = fixed_fade_distance if fixed_fade_distance > 0 else 1.0
        normalized_map = (clipped_distance_map / denominator)
    else:
        min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=receding_white_areas)
        if max_val <= min_val:
            return np.zeros_like(current_white_mask, dtype=np.uint8)
        normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

    # 5. Invert and convert to 8-bit
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

    # 6. Final mask
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)
    return final_gradient_map

def _calculate_receding_gradient_field_roi_fade(current_white_mask, prior_white_combined_mask, config, classified_rois, debug_info=None):
    """
    Calculates receding gradients on a per-ROI basis to isolate fades.
    """
    if prior_white_combined_mask is None or not classified_rois:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    final_gradient_map = np.zeros_like(current_white_mask, dtype=np.uint8)

    global_receding_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if cv2.countNonZero(global_receding_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    if debug_info:
        cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_03a_global_receding_areas.png"), global_receding_areas)

    for i, roi in enumerate(classified_rois):
        if config.roi_params.enable_raft_support_handling and roi['classification'] in ["raft", "support"]:
            if debug_info:
                print(f"Skipping ROI {roi.get('id', i)} (area: {roi['area']}), classified as {roi['classification']}.")
            continue

        roi_mask = roi['mask']

        fade_dist = int(config.fixed_fade_distance_receding)
        kernel_size = min(51, fade_dist * 2 + 1)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_roi_mask = cv2.dilate(roi_mask, kernel, iterations=1)

        roi_receding_areas = cv2.bitwise_and(global_receding_areas, dilated_roi_mask)

        if cv2.countNonZero(roi_receding_areas) == 0:
            continue

        distance_transform_src = cv2.bitwise_not(roi_mask)
        distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

        receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=roi_receding_areas)
        if np.max(receding_distance_map) == 0:
            continue

        use_fixed_normalization = config.use_fixed_fade_receding
        fixed_fade_distance = config.fixed_fade_distance_receding

        if use_fixed_normalization:
            clipped_distance_map = np.clip(receding_distance_map, 0, fixed_fade_distance)
            denominator = fixed_fade_distance if fixed_fade_distance > 0 else 1.0
            normalized_map = (clipped_distance_map / denominator)
        else:
            min_val, max_val, _, _ = cv2.minMaxLoc(receding_distance_map, mask=roi_receding_areas)
            if max_val <= min_val:
                continue
            normalized_map = (receding_distance_map - min_val) / (max_val - min_val)

        inverted_normalized_map = 1.0 - normalized_map
        roi_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

        roi_gradient_map = cv2.bitwise_and(roi_gradient_map, roi_gradient_map, mask=roi_receding_areas)

        final_gradient_map = np.maximum(final_gradient_map, roi_gradient_map)

        if debug_info:
            cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_roi_{i}_receding_areas.png"), roi_receding_areas)
            cv2.imwrite(os.path.join(debug_info['output_folder'], f"{debug_info['base_filename']}_debug_roi_{i}_gradient.png"), roi_gradient_map)

    return final_gradient_map

def _calculate_weighted_receding_gradient_field(current_white_mask, prior_binary_masks, config, debug_info=None):
    """
    Calculates a gradient field by taking a weighted sum of gradients from multiple prior layers.
    This operates in a higher bit-depth to prevent clipping during accumulation.
    """
    weights = config.manual_weights
    if not prior_binary_masks or not weights:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    num_items = min(len(prior_binary_masks), len(weights))
    if num_items == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    prior_binary_masks = prior_binary_masks[:num_items]
    weights = weights[:num_items]

    total_weight = float(sum(weights))
    if total_weight <= 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    weighted_accumulator = np.zeros(current_white_mask.shape, dtype=np.float32)

    fade_distances = config.fade_distances_receding
    default_fade_dist = config.fixed_fade_distance_receding

    # The distance map is calculated once, based on the current layer's geometry,
    # as the gradient should be smooth from the current feature's edge.
    distance_transform_src = cv2.bitwise_not(current_white_mask)
    distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    for i, (prior_mask, weight) in enumerate(zip(prior_binary_masks, weights)):
        if weight <= 0:
            continue

        fade_dist = fade_distances[i] if i < len(fade_distances) else default_fade_dist
        denominator = fade_dist if fade_dist > 0 else 1.0

        receding_white_areas = cv2.bitwise_and(prior_mask, cv2.bitwise_not(current_white_mask))
        if cv2.countNonZero(receding_white_areas) == 0:
            continue

        # Mask the single distance map with this layer's specific receding area
        receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

        clipped_distance_map = np.clip(receding_distance_map, 0, fade_dist)
        normalized_map = 1.0 - (clipped_distance_map / denominator)

        # IMPORTANT: Mask the contribution to only apply within the current prior's receding area.
        # This prevents areas outside the current receding mask from incorrectly adding a full-weighted value.
        contribution = normalized_map * weight
        weighted_accumulator += np.where(receding_white_areas > 0, contribution, 0)

    final_normalized_map = weighted_accumulator / total_weight
    final_gradient_map = (255 * final_normalized_map).astype(np.uint8)

    combined_receding_mask = np.zeros_like(current_white_mask, dtype=np.uint8)
    for prior_mask in prior_binary_masks:
        receding_area = cv2.bitwise_and(prior_mask, cv2.bitwise_not(current_white_mask))
        combined_receding_mask = cv2.bitwise_or(combined_receding_mask, receding_area)

    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=combined_receding_mask)

    return final_gradient_map


def _calculate_receding_gradient_field_enhanced_edt_scipy(receding_distance_map, labels, num_labels, fade_distance_limit):
    """
    Calculates the Enhanced EDT gradient using the SciPy vectorized approach.
    """
    max_vals_per_label = ndimage.maximum(receding_distance_map, labels=labels, index=np.arange(1, num_labels))
    max_vals_lut = np.concatenate(([0], max_vals_per_label))
    max_map = max_vals_lut[labels]

    denominator = np.minimum(max_map, fade_distance_limit)
    denominator = denominator.astype(np.float32)
    denominator[denominator <= 0] = 1.0

    clipped_map = np.clip(receding_distance_map, 0, denominator)
    normalized_map = clipped_map / denominator
    inverted_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_map).astype(np.uint8)
    return final_gradient_map

@numba.jit(nopython=True, cache=True)
def _calculate_receding_gradient_field_enhanced_edt_numba(receding_distance_map, labels, num_labels, fade_distance_limit):
    """
    Calculates the Enhanced EDT gradient using a Numba JIT-compiled loop for performance.
    """
    final_gradient_map = np.zeros_like(receding_distance_map, dtype=np.uint8)

    # First pass: find max distance for each label
    max_vals = np.zeros(num_labels, dtype=np.float32)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label > 0:
                dist = receding_distance_map[y, x]
                if dist > max_vals[label]:
                    max_vals[label] = dist

    # Second pass: calculate final gradient
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = labels[y, x]
            if label > 0:
                dist = receding_distance_map[y, x]

                denominator = min(max_vals[label], fade_distance_limit)
                if denominator > 0:
                    clipped_dist = min(dist, denominator)
                    normalized = clipped_dist / denominator
                    inverted = 1.0 - normalized
                    final_gradient_map[y, x] = int(inverted * 255)

    return final_gradient_map

def _calculate_receding_gradient_field_enhanced_edt(current_white_mask, prior_binary_masks, config, debug_info=None):
    """
    Dispatcher for Enhanced EDT. Sets up common data then calls either the
    SciPy or Numba implementation based on the user's config.
    """
    if not prior_binary_masks:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    prior_white_combined_mask = find_prior_combined_white_mask(prior_binary_masks)
    if prior_white_combined_mask is None:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    distance_transform_src = cv2.bitwise_not(current_white_mask)

    # --- Anisotropic Correction ---
    if config.anisotropic_params.enabled:
        ap = config.anisotropic_params
        original_height, original_width = distance_transform_src.shape

        # Clamp factors to prevent excessive scaling
        x_factor = max(0.1, min(10.0, ap.x_factor))
        y_factor = max(0.1, min(10.0, ap.y_factor))

        # Only resize if factors are not 1.0 to avoid unnecessary work
        if x_factor != 1.0 or y_factor != 1.0:
            new_width = int(original_width * x_factor)
            new_height = int(original_height * y_factor)

            # Resize the source mask, calculate distance, then resize back
            resized_src = cv2.resize(distance_transform_src, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
            resized_dist_map = cv2.distanceTransform(resized_src, cv2.DIST_L2, 5)

            # Resize the distance map back to original dimensions
            distance_map = cv2.resize(resized_dist_map, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        else:
            distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)
    else:
        distance_map = cv2.distanceTransform(distance_transform_src, cv2.DIST_L2, 5)

    receding_distance_map = cv2.bitwise_and(distance_map, distance_map, mask=receding_white_areas)

    num_labels, labels = cv2.connectedComponents(receding_white_areas)
    if num_labels <= 1:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    fade_distance_limit = config.fixed_fade_distance_receding

    if config.use_numba_jit:
        try:
            final_gradient_map = _calculate_receding_gradient_field_enhanced_edt_numba(
                receding_distance_map, labels.astype(np.int32), num_labels, fade_distance_limit
            )
        except Exception as e:
            print(f"Numba JIT execution failed: {e}. Falling back to SciPy implementation.")
            final_gradient_map = _calculate_receding_gradient_field_enhanced_edt_scipy(
                receding_distance_map, labels, num_labels, fade_distance_limit
            )
    else:
        final_gradient_map = _calculate_receding_gradient_field_enhanced_edt_scipy(
            receding_distance_map, labels, num_labels, fade_distance_limit
        )

    return cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)


def _calculate_receding_gradient_field_tiledb_3d_aa(current_white_mask, layer_index, tiledb_uri, config, debug_info=None):
    """
    Calculates a gradient using orthogonal XZ and YZ slices from a TileDB store
    for anti-aliasing.
    """
    if not tiledb_uri or not config.use_tiledb:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # 1. Find the edges of the current layer. This is where we'll apply AA.
    edges = cv2.Canny(current_white_mask, 100, 200)
    edge_coords = np.argwhere(edges > 0) # Returns list of [y, x]

    if edge_coords.size == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # Create an output gradient map to populate
    gradient_map = np.zeros_like(current_white_mask, dtype=np.uint8)

    # For this simple example, we'll use a small kernel to average around the current pixel
    # in the orthogonal slices. A kernel size of 3 means we look one pixel "up/down"
    # and "left/right" in the orthogonal planes.
    kernel_size = 3
    half_kernel = kernel_size // 2

    # 2. For each edge pixel, get its orthogonal slices and calculate the new value.
    with tiledb.open(tiledb_uri, 'r') as A:
        num_layers, height, width = A.shape

        for y, x in edge_coords:
            # Get the pixel values in the column above and below this pixel across all layers
            yz_slice = A[:, :, x]['pixel_value'] # Shape: (num_layers, height)

            # Get the pixel values in the row left and right of this pixel across all layers
            xz_slice = A[:, y, :]['pixel_value'] # Shape: (num_layers, width)

            # --- YZ Slice (Vertical) ---
            # Define the bounds for the kernel in the YZ slice (around the current layer)
            z_start = max(0, layer_index - half_kernel)
            z_end = min(num_layers, layer_index + half_kernel + 1)
            y_start = max(0, y - half_kernel)
            y_end = min(height, y + half_kernel + 1)

            yz_kernel_values = yz_slice[z_start:z_end, y_start:y_end]

            # --- XZ Slice (Horizontal) ---
            # Define the bounds for the kernel in the XZ slice (around the current layer)
            z_start = max(0, layer_index - half_kernel)
            z_end = min(num_layers, layer_index + half_kernel + 1)
            x_start = max(0, x - half_kernel)
            x_end = min(width, x + half_kernel + 1)

            xz_kernel_values = xz_slice[z_start:z_end, x_start:x_end]

            # Combine the values and average them
            # This is a very basic filter. More complex weighting could be used.
            all_values = np.concatenate((yz_kernel_values.flatten(), xz_kernel_values.flatten()))

            if all_values.size > 0:
                avg_value = np.mean(all_values)
                gradient_map[y, x] = int(avg_value)

    return gradient_map

def process_z_blending(current_white_mask, prior_masks, config, classified_rois, debug_info=None, layer_index=None, tiledb_uri=None):
    """
    Main entry point for Z-axis blending. Dispatches to the correct blending mode.
    `prior_masks` can be a single combined mask or a list of masks depending on the mode.
    """
    if config.blending_mode == ProcessingMode.TILEDB_3D_AA:
        return _calculate_receding_gradient_field_tiledb_3d_aa(
            current_white_mask,
            layer_index,
            tiledb_uri,
            config,
            debug_info
        )
    elif config.blending_mode == ProcessingMode.ROI_FADE:
        return _calculate_receding_gradient_field_roi_fade(
            current_white_mask,
            prior_masks,
            config,
            classified_rois,
            debug_info
        )
    elif config.blending_mode == ProcessingMode.WEIGHTED_STACK:
        return _calculate_weighted_receding_gradient_field(
            current_white_mask,
            prior_masks,
            config,
            debug_info
        )
    elif config.blending_mode == ProcessingMode.ENHANCED_EDT:
        return _calculate_receding_gradient_field_enhanced_edt(
            current_white_mask,
            prior_masks,
            config,
            debug_info
        )
    else:  # Default to FIXED_FADE
        return _calculate_receding_gradient_field_fixed_fade(
            current_white_mask,
            prior_masks,
            config,
            debug_info
        )

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