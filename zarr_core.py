import numpy as np
import numba
import zarr
from scipy import ndimage
from config import Config, ProcessingMode

def _calculate_receding_gradient_field_enhanced_edt_zarr(binary_stack, config: Config):
    """
    Calculates the Enhanced EDT gradient for a full 3D Zarr stack.
    This is the 3D adaptation of the original 2D processing_core function.
    """
    # 1. Create a shifted version of the stack to find prior-layer white areas
    # Shift the array down by one along the z-axis (axis 0)
    # The last slice is filled with 0 (black) as it has no prior layer in this context
    prior_white_stack = np.zeros_like(binary_stack)
    prior_white_stack[1:] = binary_stack[:-1]

    # 2. Identify "receding white areas" in 3D
    # These are voxels that are black (0) in the current layer but were white (255) in the prior layer.
    receding_white_areas = (prior_white_stack == 255) & (binary_stack == 0)

    if not np.any(receding_white_areas):
        return np.zeros_like(binary_stack, dtype=np.uint8)

    # 3. Calculate 3D Euclidean Distance Transform
    # We want the distance from any point to the nearest solid (white) feature.
    # The input to distance_transform_edt is a binary image where non-zero values are considered features.
    # We invert the binary stack so that solid features are 0 and empty space is 1.
    # The transform then gives the distance from any non-zero pixel to the nearest zero pixel.
    distance_transform_src = (binary_stack == 0).astype(np.uint8)

    # The `sampling` parameter will be used for anisotropy later. For now, it's isotropic.
    if config.anisotropic_flags.edt_enabled:
        sampling = [config.voxel_dim_z, config.voxel_dim_y, config.voxel_dim_x]
    else:
        sampling = None # Isotropic

    distance_map = ndimage.distance_transform_edt(distance_transform_src, sampling=sampling)

    # 4. Mask the distance map to only include the receding areas
    receding_distance_map = distance_map * receding_white_areas

    # 5. Identify connected components (regions) in the receding areas
    # `ndimage.label` is the 3D equivalent of cv2.connectedComponents
    labels, num_labels = ndimage.label(receding_white_areas)
    if num_labels == 0:
        return np.zeros_like(binary_stack, dtype=np.uint8)

    # 6. Normalize the gradient within each component
    # This is the "Enhanced" part of the EDT, where each region's fade is based on its own size.
    fade_distance_limit = config.fixed_fade_distance_receding

    # Find the maximum distance value within each labeled region
    # The `index` parameter makes this operation very efficient.
    max_vals_per_label = ndimage.maximum(receding_distance_map, labels=labels, index=np.arange(1, num_labels + 1))

    # Create a lookup table for the max values. Label 0 (background) has a max_val of 0.
    max_vals_lut = np.concatenate(([0], max_vals_per_label))

    # Create a full 3D map of the max values by indexing the LUT with the labels array.
    max_map = max_vals_lut[labels]

    # The denominator for normalization is the smaller of the region's max distance or the user-defined limit.
    denominator = np.minimum(max_map, fade_distance_limit)
    denominator = denominator.astype(np.float32)
    denominator[denominator <= 0] = 1.0 # Avoid division by zero

    # Clip the distance map to the calculated denominator
    clipped_map = np.clip(receding_distance_map, 0, denominator)

    # Normalize the map (0.0 to 1.0)
    normalized_map = clipped_map / denominator

    # Invert the map so that areas closest to the feature edge are brightest (1.0)
    inverted_map = 1.0 - normalized_map

    # Scale to 8-bit integer range (0-255)
    final_gradient_map = (255 * inverted_map).astype(np.uint8)

    # 7. Final mask to ensure we only have values in the receding areas
    return final_gradient_map * receding_white_areas

def process_z_blending_zarr(input_stack: zarr.Array, config: Config):
    """
    Main entry point for Z-axis blending on a Zarr stack.
    """
    print("zarr_core.py: process_z_blending_zarr called with 3D-aware logic.")

    # Create a binary version of the stack (0 or 255) for calculations
    # This is done in memory. For very large stacks, this could be chunked.
    binary_stack = (input_stack[:] > 127).astype(np.uint8) * 255

    # Dispatch to the correct blending function based on config
    # For now, we only have the Enhanced EDT implementation for Zarr
    if config.blending_mode == ProcessingMode.ENHANCED_EDT:
        gradient_field = _calculate_receding_gradient_field_enhanced_edt_zarr(binary_stack, config)
    else:
        # Placeholder for other modes
        print(f"Warning: Blending mode '{config.blending_mode.value}' not yet implemented for Zarr. Returning empty field.")
        gradient_field = np.zeros_like(binary_stack, dtype=np.uint8)

    # Merge the gradient field back into the original data
    # The gradient should only fill in the black areas (0) of the original image.
    # Using np.maximum achieves this perfectly.
    output_data = np.maximum(input_stack[:], gradient_field)

    # Create an output Zarr array
    output_stack = zarr.array(output_data, chunks=input_stack.chunks, dtype=input_stack.dtype)

    return output_stack
