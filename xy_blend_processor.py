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

# xy_blend_processor.py (Modified)

"""
Processes an image through a sequence of XY blending, smoothing, sharpening,
resizing, and LUT application operations as defined by the provided pipeline.
"""

import cv2
import numpy as np
import os
from typing import List, Any, Optional # Removed Dict as we're using dataclasses
import lut_manager # Import lut_manager for LUT generation and application
from config import XYBlendOperation, LutParameters # Import the new dataclasses

# Removed:
# _config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now
# def set_config_reference(config_instance: Any):
#     """Sets the reference to the global Config instance."""
#     global _config_ref
#     _config_ref = config_instance

def _ensure_odd_ksize(ksize: int) -> int:
    """Ensures a kernel size is an odd integer, adjusting if necessary."""
    if ksize % 2 == 0:
        return ksize + 1 if ksize > 0 else 1
    return ksize

def apply_gaussian_blur(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Applies Gaussian blur to an 8-bit grayscale image.
    Uses configurable separable X and Y kernel sizes and sigmas.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize_x = op.gaussian_ksize_x # Already ensured odd by __post_init__ in Config
    ksize_y = op.gaussian_ksize_y # Already ensured odd by __post_init__ in Config
    sigma_x = op.gaussian_sigma_x
    sigma_y = op.gaussian_sigma_y

    # cv2.GaussianBlur expects ksize as a tuple (width, height)
    return cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

def apply_bilateral_filter(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Applies a bilateral filter to an 8-bit grayscale image.
    Effective for noise reduction while preserving edges.
    """
    # Access attributes directly from the XYBlendOperation object
    d = op.bilateral_d
    sigma_color = op.bilateral_sigma_color
    sigma_space = op.bilateral_sigma_space

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

def apply_median_blur(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Applies a median blur to an 8-bit grayscale image.
    Excellent for removing salt-and-pepper noise.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize = op.median_ksize # Already ensured odd by __post_init__ in Config
    if ksize <= 1: # Median blur requires kernel size > 1
        return image # No-op if kernel size is 1 or invalid
    return cv2.medianBlur(image, ksize)

def apply_unsharp_mask(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Applies unsharp masking to an 8-bit grayscale image for sharpening.
    """
    # Access attributes directly from the XYBlendOperation object
    amount = op.unsharp_amount
    threshold = op.unsharp_threshold
    blur_ksize = op.unsharp_blur_ksize # Already ensured odd by __post_init__ in Config
    blur_sigma = op.unsharp_blur_sigma

    # Create a blurred version of the image
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)

    # Calculate the 'detail' layer (original - blurred)
    # Use float32 for intermediate calculation to avoid clipping negative values
    detail_layer = image.astype(np.float32) - blurred_image.astype(np.float32)

    # Apply threshold to the detail layer
    # Pixels where original intensity is below threshold are not sharpened
    if threshold > 0:
        # Create a mask where detail should be applied
        mask = (image > threshold).astype(np.float32)
        detail_layer *= mask # Apply mask to detail layer

    # Add the scaled detail back to the original image
    # Use addWeighted for controlled blending and automatic clipping to 0-255
    sharpened_image = cv2.addWeighted(image, 1.0 + amount, blurred_image, -amount, 0)
    
    # Ensure final output is uint8 and clipped to 0-255
    return np.clip(sharpened_image, 0, 255).astype(np.uint8)

def apply_resize(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Resizes an image to the specified width and height using the given resampling method.
    """
    # Access attributes directly from the XYBlendOperation object
    width = op.resize_width
    height = op.resize_height
    resample_mode = op.resample_mode.upper()

    if width is None and height is None:
        return image # No resize requested

    current_height, current_width = image.shape[:2]

    # Handle cases where only one dimension is specified, maintaining aspect ratio
    if width is None:
        if height is not None:
            width = int(current_width * (height / current_height))
        else: # Both are None
            return image
    elif height is None:
        if width is not None:
            height = int(current_height * (width / current_width))
        else: # Both are None
            return image
    
    # If target dimensions are the same as current, no need to resize
    if width == current_width and height == current_height:
        return image

    flags = {
        "NEAREST": cv2.INTER_NEAREST,
        "BILINEAR": cv2.INTER_LINEAR,
        "BICUBIC": cv2.INTER_CUBIC,
        "LANCZOS4": cv2.INTER_LANCZOS4,
        "AREA": cv2.INTER_AREA
    }
    interp = flags.get(resample_mode, cv2.INTER_LANCZOS4) # Default to LANCZOS4

    return cv2.resize(image, (width, height), interpolation=interp)

def apply_lut_operation(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    """
    Generates/loads a LUT based on operation parameters and applies it to the image.
    """
    lut_params = op.lut_params
    generated_lut: Optional[np.ndarray] = None

    if lut_params.lut_source == "generated":
        try:
            if lut_params.lut_generation_type == "linear":
                generated_lut = lut_manager.generate_linear_lut(lut_params.linear_min_input, lut_params.linear_max_output)
            elif lut_params.lut_generation_type == "gamma":
                generated_lut = lut_manager.generate_gamma_lut(lut_params.gamma_value)
            elif lut_params.lut_generation_type == "s_curve":
                generated_lut = lut_manager.generate_s_curve_lut(lut_params.s_curve_contrast)
            elif lut_params.lut_generation_type == "log":
                generated_lut = lut_manager.generate_log_lut(lut_params.log_param)
            elif lut_params.lut_generation_type == "exp":
                generated_lut = lut_manager.generate_exp_lut(lut_params.exp_param)
            elif lut_params.lut_generation_type == "sqrt":
                generated_lut = lut_manager.generate_sqrt_lut(lut_params.sqrt_param)
            elif lut_params.lut_generation_type == "rodbard":
                generated_lut = lut_manager.generate_rodbard_lut(lut_params.rodbard_param)
            else:
                print(f"Warning: Unknown LUT generation type '{lut_params.lut_generation_type}'. Using default linear LUT.")
                generated_lut = lut_manager.get_default_z_lut()
        except ValueError as e:
            print(f"Error generating LUT for operation '{op.type}' ({lut_params.lut_generation_type}): {e}. Using default linear LUT.")
            generated_lut = lut_manager.get_default_z_lut()
        except Exception as e:
            print(f"An unexpected error occurred during LUT generation for operation '{op.type}': {e}. Using default linear LUT.")
            generated_lut = lut_manager.get_default_z_lut()

    elif lut_params.lut_source == "file":
        if lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
            try:
                generated_lut = lut_manager.load_lut(lut_params.fixed_lut_path)
            except Exception as e:
                print(f"Error loading LUT from file '{lut_params.fixed_lut_path}' for operation '{op.type}': {e}. Using default linear LUT.")
                generated_lut = lut_manager.get_default_z_lut()
        else:
            print(f"Warning: LUT file '{lut_params.fixed_lut_path}' not found for operation '{op.type}'. Using default linear LUT.")
            generated_lut = lut_manager.get_default_z_lut()
    else:
        print(f"Warning: Unknown LUT source '{lut_params.lut_source}' for operation '{op.type}'. Using default linear LUT.")
        generated_lut = lut_manager.get_default_z_lut()

    if generated_lut is None:
        print(f"Error: No LUT could be determined for operation '{op.type}'. Skipping LUT application.")
        return image # Return original image if LUT generation/load fails

    return lut_manager.apply_z_lut(image, generated_lut)


def process_xy_pipeline(image: np.ndarray, pipeline_ops: List[XYBlendOperation]) -> np.ndarray:
    """
    Applies the sequence of XY blending/processing operations defined in the pipeline_ops list.
    The input image is expected to be an 8-bit grayscale NumPy array.
    """
    processed_image = image.copy()

    if not pipeline_ops:
        return processed_image # Return original if no ops

    for op in pipeline_ops:
        op_type = op.type # Access the type directly from the XYBlendOperation object

        if op_type == "none":
            continue # Skip no-op
        elif op_type == "gaussian_blur":
            processed_image = apply_gaussian_blur(processed_image, op)
        elif op_type == "bilateral_filter":
            processed_image = apply_bilateral_filter(processed_image, op)
        elif op_type == "median_blur":
            processed_image = apply_median_blur(processed_image, op)
        elif op_type == "unsharp_mask":
            processed_image = apply_unsharp_mask(processed_image, op)
        elif op_type == "resize":
            processed_image = apply_resize(processed_image, op)
        elif op_type == "apply_lut":
            processed_image = apply_lut_operation(processed_image, op)
        else:
            print(f"Warning: Unknown XY blend operation type '{op_type}'. Skipping.")
        
        # Ensure image remains 8-bit after each operation (most OpenCV ops do this if input is uint8)
        if processed_image.dtype != np.uint8:
            processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)
    
    return processed_image

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- XY Blend Processor Module Test ---")
    
    # Create a dummy 8-bit grayscale image (e.g., 100x100 with some noise/features)
    test_image = np.zeros((100, 100), dtype=np.uint8)
    test_image[20:40, 20:40] = 150 # A square
    test_image[60:80, 60:80] = 200 # Another square
    test_image = cv2.randn(test_image, (0), (20)) # Add some random noise
    test_image = np.clip(test_image, 0, 255).astype(np.uint8)
    cv2.imwrite("test_xy_original.png", test_image)
    print("Original test image saved as test_xy_original.png")

    # Import actual classes for testing from the new config
    from config import Config, XYBlendOperation, LutParameters 

    # Test pipeline: Gaussian Blur -> Apply LUT (gamma) -> Resize -> Unsharp Mask
    print("\nTesting pipeline: Gaussian Blur -> Apply LUT (gamma) -> Resize -> Unsharp Mask")
    test_pipeline_ops = [
        XYBlendOperation(type="gaussian_blur", gaussian_ksize_x=5, gaussian_ksize_y=5, gaussian_sigma_x=1.0, gaussian_sigma_y=1.0),
        XYBlendOperation(type="apply_lut", lut_params=LutParameters(lut_source="generated", lut_generation_type="gamma", gamma_value=0.7)),
        XYBlendOperation(type="resize", resize_width=50, resize_height=50, resample_mode="BICUBIC"),
        XYBlendOperation(type="unsharp_mask", unsharp_amount=1.5, unsharp_threshold=10, unsharp_blur_ksize=3)
    ]

    output_image_pipeline = process_xy_pipeline(test_image, test_pipeline_ops)
    cv2.imwrite("test_xy_pipeline_output.png", output_image_pipeline)
    print(f"Pipeline output shape: {output_image_pipeline.shape}, dtype: {output_image_pipeline.dtype}")
    print("Pipeline output saved as test_xy_pipeline_output.png")

    # Test individual LUT operation (e.g., loading from a dummy file)
    print("\nTesting individual LUT operation (file-based):")
    # First, create a dummy LUT file for testing
    dummy_lut_file = "test_lut.json"
    dummy_lut = lut_manager.generate_linear_lut(50, 200)
    lut_manager.save_lut(dummy_lut_file, dummy_lut)
    print(f"Dummy LUT saved to {dummy_lut_file}")

    file_lut_op = XYBlendOperation(type="apply_lut", lut_params=LutParameters(lut_source="file", fixed_lut_path=dummy_lut_file))
    output_file_lut = process_xy_pipeline(test_image, [file_lut_op])
    cv2.imwrite("test_xy_file_lut_output.png", output_file_lut)
    print("File-based LUT output saved as test_xy_file_lut_output.png")

    # Clean up test images
    import os
    for f in ["test_xy_original.png", "test_xy_pipeline_output.png", "test_xy_file_lut_output.png", dummy_lut_file]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")