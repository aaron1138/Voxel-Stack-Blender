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
import skfmm

from config import Config

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

    # Invert (so edges are white and it fades to black) and scale to 8-bit (0-255)
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_chunk = (255 * inverted_normalized_map).astype(np.uint8)

    return final_gradient_chunk

def process_sdf_3d_fmm_window(image_chunk, config: Config):
    """
    Processes a chunk of images using the Fast Marching Method from scikit-fmm.
    """
    params = config.sdf_3d_params
    # Note: skfmm uses dx, dy, dz order, which corresponds to our Z, Y, X voxel dimensions.
    voxel_dims = (params.voxel_dim_z, params.voxel_dim_y, params.voxel_dim_x)

    # skfmm works on a level set. Positive values are outside, negative are inside.
    # The zero-contour is the surface. We create this by subtracting 0.5 from the 0/1 array.
    level_set = image_chunk - 0.5

    # Calculate the signed distance field using the Fast Marching Method.
    # This is a multi-threaded operation.
    distance_map_microns = skfmm.distance(level_set, dx=voxel_dims)

    # The rest of the logic is identical to the SciPy version.
    if params.radius_unit == "pixels":
        avg_pixel_dim = (params.voxel_dim_x + params.voxel_dim_y + params.voxel_dim_z) / 3.0
        gradient_radius_microns = params.gradient_radius * avg_pixel_dim
    else: # Microns
        gradient_radius_microns = params.gradient_radius

    if gradient_radius_microns <= 0:
        gradient_radius_microns = 1.0

    # Clip the distance map (we only care about the distance outside the object, which is positive)
    clipped_distance_map = np.clip(distance_map_microns, 0, gradient_radius_microns)

    # Normalize, invert, and scale
    normalized_map = clipped_distance_map / gradient_radius_microns
    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_chunk = (255 * inverted_normalized_map).astype(np.uint8)

    return final_gradient_chunk
