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
from config import XYBlendOperation, LutParameters

def _anisotropic_resize(image, config, invert=False):
    """Helper to resize image for anisotropic correction."""
    if config.voxel_x_um == config.voxel_y_um:
        return image # No correction needed if isotropic

    min_dim = min(config.voxel_x_um, config.voxel_y_um)
    x_factor = min_dim / config.voxel_x_um
    y_factor = min_dim / config.voxel_y_um

    original_height, original_width = image.shape[:2]

    if invert:
        # When inverting, the target is the original size
        new_width = original_width
        new_height = original_height
        # Use a smoother interpolation when resizing back up
        interp = cv2.INTER_LINEAR
    else:
        # Resize to make pixels isotropic
        new_width = int(original_width * x_factor)
        new_height = int(original_height * y_factor)
        # Use area-based interpolation for downsampling
        interp = cv2.INTER_AREA

    return cv2.resize(image, (new_width, new_height), interpolation=interp)

def apply_gaussian_blur(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
    """Applies Gaussian blur to an 8-bit grayscale image."""
    ksize_x = op.gaussian_ksize_x
    ksize_y = op.gaussian_ksize_y
    sigma_x = op.gaussian_sigma_x
    sigma_y = op.gaussian_sigma_y

    if op.anisotropic_correction_enabled:
        resized_image = _anisotropic_resize(image, config)
        # When pixels are isotropic, use the same sigma for both axes, e.g., the average
        avg_sigma = (sigma_x + sigma_y) / 2
        blurred_resized = cv2.GaussianBlur(resized_image, (ksize_x, ksize_y), sigmaX=avg_sigma, sigmaY=avg_sigma)
        return _anisotropic_resize(blurred_resized, config, invert=True)
    else:
        return cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

def apply_bilateral_filter(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
    """Applies a bilateral filter to an 8-bit grayscale image."""
    d = op.bilateral_d
    sigma_color = op.bilateral_sigma_color
    sigma_space = op.bilateral_sigma_space

    if op.anisotropic_correction_enabled:
        resized_image = _anisotropic_resize(image, config)
        # Sigma space is a distance, so it should be applied to the isotropically-scaled image
        filtered_resized = cv2.bilateralFilter(resized_image, d, sigma_color, sigma_space)
        return _anisotropic_resize(filtered_resized, config, invert=True)
    else:
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_median_blur(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
    """Applies a median blur to an 8-bit grayscale image."""
    ksize = op.median_ksize
    if ksize <= 1: return image
    return cv2.medianBlur(image, ksize)

def apply_unsharp_mask(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
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

def apply_resize(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
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

def apply_lut_operation(image: np.ndarray, op: XYBlendOperation, config: "Config") -> np.ndarray:
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


def process_xy_pipeline(image: np.ndarray, pipeline_ops: List[XYBlendOperation], config: "Config") -> np.ndarray:
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
            processed_image = op_func(processed_image, op, config)
        elif op.type != "none":
            print(f"Warning: Unknown XY blend operation type '{op.type}'. Skipping.")
        
        # It's good practice to ensure the image remains 8-bit after each step,
        # although most OpenCV functions handle this correctly if the input is uint8.
        if processed_image.dtype != np.uint8:
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    return processed_image
