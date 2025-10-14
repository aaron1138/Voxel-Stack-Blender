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

import numpy as np
from scipy import ndimage
import cv2

from config import Config
import lut_manager

def process_sdf_3d_window(image_chunk, config: Config):
    """
    Processes a chunk of images using a 3D Signed Distance Field.

    Args:
        image_chunk (np.ndarray): A 3D numpy array of shape (depth, height, width),
                                  representing the sliding window of binary images.
        config (Config): The application configuration.

    Returns:
        np.ndarray: A 3D numpy array of the same shape, containing the processed
                    grayscale gradients for the central part of the chunk.
    """
    params = config.sdf_3d_params
    voxel_dims = (params.voxel_dim_z, params.voxel_dim_y, params.voxel_dim_x)

    # Invert the chunk because distance transform finds distance to nearest ZERO
    inverted_chunk = np.logical_not(image_chunk).astype(np.uint8)

    # Calculate the anisotropic distance transform
    # The result is the true distance in micrometers from each voxel to the nearest feature edge.
    distance_map_microns = ndimage.distance_transform_edt(inverted_chunk, sampling=voxel_dims)

    # Normalize the gradient based on the user-defined radius
    if params.radius_unit == "pixels":
        # Convert pixel radius to an effective micrometer radius
        avg_pixel_dim = (params.voxel_dim_x + params.voxel_dim_y + params.voxel_dim_z) / 3.0
        gradient_radius_microns = params.gradient_radius * avg_pixel_dim
    else: # Microns
        gradient_radius_microns = params.gradient_radius

    if gradient_radius_microns <= 0:
        gradient_radius_microns = 1.0 # Avoid division by zero

    # Clip the distance map to the desired radius
    clipped_distance_map = np.clip(distance_map_microns, 0, gradient_radius_microns)

    # Normalize to a 0-1 range
    normalized_map = clipped_distance_map / gradient_radius_microns

    # Invert (so edges are white and it fades to black)
    inverted_normalized_map = 1.0 - normalized_map

    # --- Apply non-linear normalization if enabled (THE CORRECT, HIGH-PRECISION WAY) ---
    if params.enable_non_linear_norm:
        # Get the mathematical function (e.g., a spline object or a lambda for gamma)
        curve_function = lut_manager.get_curve_function_from_params(params.non_linear_params)

        # Apply the function directly to the high-precision float array
        final_float_map = curve_function(inverted_normalized_map)
    else:
        # If not enabled, the map remains linear
        final_float_map = inverted_normalized_map

    # Convert to 8-bit as the very last step
    final_gradient_chunk = (255 * np.clip(final_float_map, 0.0, 1.0)).astype(np.uint8)

    return final_gradient_chunk
