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

# xy_blend_processor.py (Completed)

import cv2
import numpy as np
import os
from typing import List, Optional
import lut_manager
from config import XYBlendOperation, LutParameters, app_config

def _apply_anisotropic_blur(image: np.ndarray, blur_function, *args, **kwargs):
    """
    Wrapper to apply a blur function with anisotropic correction.
    Resizes the image to be isotropic, applies the blur, and resizes back.
    """
    if app_config.voxel_x_um <= 0 or app_config.voxel_y_um <= 0:
        return blur_function(image, *args, **kwargs)

    aspect_ratio = app_config.voxel_y_um / app_config.voxel_x_um
    if abs(aspect_ratio - 1.0) < 0.01:
        return blur_function(image, *args, **kwargs)

    original_height, original_width = image.shape
    new_width = int(original_width * aspect_ratio)

    resized_image = cv2.resize(image, (new_width, original_height), interpolation=cv2.INTER_LINEAR)

    # We need to adjust the kernel size for the resized dimension
    # For simplicity, we can scale the kernel width, or use an average kernel size.
    # Let's adjust the kernel width in kwargs if it exists.
    new_kwargs = kwargs.copy()
    if 'ksize' in new_kwargs and isinstance(new_kwargs['ksize'], tuple):
        kx, ky = new_kwargs['ksize']
        new_kx = int(kx * aspect_ratio)
        new_kx = new_kx if new_kx % 2 != 0 else new_kx + 1 # Ensure odd
        new_kwargs['ksize'] = (new_kx, ky)

    blurred_resized = blur_function(resized_image, *args, **new_kwargs)

    return cv2.resize(blurred_resized, (original_width, original_height), interpolation=cv2.INTER_LINEAR)


def apply_gaussian_blur(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Applies Gaussian blur to an 8-bit grayscale image."""
    ksize = (op.gaussian_ksize_x, op.gaussian_ksize_y)
    sigma_x = op.gaussian_sigma_x
    sigma_y = op.gaussian_sigma_y

    if op.anisotropic_correction:
        return _apply_anisotropic_blur(image, cv2.GaussianBlur, ksize=ksize, sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_DEFAULT)
    else:
        return cv2.GaussianBlur(image, ksize, sigmaX=sigma_x, sigmaY=sigma_y)

def apply_bilateral_filter(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Applies a bilateral filter to an 8-bit grayscale image."""
    d = op.bilateral_d
    sigma_color = op.bilateral_sigma_color
    sigma_space = op.bilateral_sigma_space

    if op.anisotropic_correction:
        # Note: Anisotropic correction for bilateral filter is complex.
        # Resizing before and after is a simplification and may not be physically accurate.
        # The 'sigmaSpace' parameter is the one affected by geometric distortion.
        return _apply_anisotropic_blur(image, cv2.bilateralFilter, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)
    else:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)


def apply_median_blur(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Applies a median blur to an 8-bit grayscale image."""
    ksize = op.median_ksize
    if ksize <= 1: return image

    if op.anisotropic_correction:
        # For median blur, we need to adjust the kernel size for the anisotropic dimension.
        if app_config.voxel_x_um > 0 and app_config.voxel_y_um > 0:
            aspect_ratio = app_config.voxel_y_um / app_config.voxel_x_um
            if abs(aspect_ratio - 1.0) > 0.01:
                # This is a simplification. A proper anisotropic median filter is non-trivial.
                # We will use an elliptical kernel, but OpenCV's medianBlur only supports square kernels.
                # As a proxy, we resize, blur with original kernel, and resize back.
                return _apply_anisotropic_blur(image, cv2.medianBlur, ksize=ksize)
        return cv2.medianBlur(image, ksize) # Fallback if no correction needed
    else:
        return cv2.medianBlur(image, ksize)

def apply_unsharp_mask(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Applies unsharp masking to an 8-bit grayscale image for sharpening."""
    amount = op.unsharp_amount
    threshold = op.unsharp_threshold
    blur_ksize = op.unsharp_blur_ksize
    blur_sigma = op.unsharp_blur_sigma

    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)
    
    # Add the weighted "detail" layer back to the original
    sharpened_image = cv2.addWeighted(image, 1.0 + amount, blurred_image, -amount, 0)
    
    if threshold > 0:
        # Create a mask where the absolute difference between original and sharpened is below the threshold
        diff = cv2.absdiff(image, sharpened_image)
        mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        # Where the mask is white (difference is small), use the original image pixels
        return np.where(mask == 255, image, sharpened_image)
    
    return sharpened_image

def apply_resize(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Resizes an image using the specified parameters."""
    width = op.resize_width
    height = op.resize_height
    if width is None and height is None: return image

    current_height, current_width = image.shape[:2]
    if width is None:
        if height is None: return image # Should not happen, but for safety
        width = int(current_width * (height / current_height))
    elif height is None:
        height = int(current_height * (width / current_width))
    
    if width == current_width and height == current_height: return image

    flags = {
        "NEAREST": cv2.INTER_NEAREST, "BILINEAR": cv2.INTER_LINEAR,
        "BICUBIC": cv2.INTER_CUBIC, "LANCZOS4": cv2.INTER_LANCZOS4,
        "AREA": cv2.INTER_AREA
    }
    interp = flags.get(op.resample_mode.upper(), cv2.INTER_LANCZOS4)
    return cv2.resize(image, (width, height), interpolation=interp)

def apply_lut_operation(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """Generates/loads a LUT based on operation parameters and applies it."""
    lut_params = op.lut_params
    generated_lut: Optional[np.ndarray] = None

    try:
        if lut_params.lut_source == "generated":
            # Pass universal range parameters to all generation functions
            args = (lut_params.input_min, lut_params.input_max, lut_params.output_min, lut_params.output_max)
            
            if lut_params.lut_generation_type == "linear":
                generated_lut = lut_manager.generate_linear_lut(*args)
            elif lut_params.lut_generation_type == "gamma":
                generated_lut = lut_manager.generate_gamma_lut(lut_params.gamma_value, *args)
            elif lut_params.lut_generation_type == "s_curve":
                generated_lut = lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast, *args)
            elif lut_params.lut_generation_type == "log":
                generated_lut = lut_manager.generate_log_lut(lut_params.log_param, *args)
            elif lut_params.lut_generation_type == "exp":
                generated_lut = lut_manager.generate_exp_lut(lut_params.exp_param, *args)
            elif lut_params.lut_generation_type == "sqrt":
                generated_lut = lut_manager.generate_sqrt_lut(lut_params.sqrt_param, *args)
            elif lut_params.lut_generation_type == "rodbard":
                generated_lut = lut_manager.generate_rodbard_lut(lut_params.rodbard_param, *args)
            elif lut_params.lut_generation_type == "spline":
                generated_lut = lut_manager.generate_spline_lut(lut_params.spline_points, *args)
        
        elif lut_params.lut_source == "file":
            if lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
                generated_lut = lut_manager.load_lut(lut_params.fixed_lut_path)
            else:
                print(f"Warning: LUT file not found: '{lut_params.fixed_lut_path}'. Using default pass-through LUT.")
    
    except Exception as e:
        print(f"Error processing LUT for operation '{op.type}': {e}. Using default pass-through LUT.")

    # If LUT generation or loading failed for any reason, use a default pass-through LUT
    if generated_lut is None:
        generated_lut = lut_manager.get_default_z_lut()

    return lut_manager.apply_z_lut(image, generated_lut)


def process_xy_pipeline(image: np.ndarray, pipeline_ops: List[XYBlendOperation]) -> np.ndarray:
    """Applies the sequence of operations defined in the pipeline."""
    processed_image = image.copy()

    if not pipeline_ops:
        return processed_image

    # Map operation type strings to their corresponding functions for clean dispatch
    op_map = {
        "gaussian_blur": apply_gaussian_blur,
        "bilateral_filter": apply_bilateral_filter,
        "median_blur": apply_median_blur,
        "unsharp_mask": apply_unsharp_mask,
        "resize": apply_resize,
        "apply_lut": apply_lut_operation,
    }

    for op in pipeline_ops:
        op_func = op_map.get(op.type)
        if op_func:
            processed_image = op_func(processed_image, op)
        elif op.type != "none":
            print(f"Warning: Unknown XY blend operation type '{op.type}'. Skipping.")
        
        # It's good practice to ensure the image remains 8-bit after each step,
        # although most OpenCV functions handle this correctly if the input is uint8.
        if processed_image.dtype != np.uint8:
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    return processed_image
