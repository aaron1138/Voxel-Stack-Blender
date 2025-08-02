# stacking_processor.py - Gemini messing with the cast to unit8 and inits.

"""
The core image processing engine for Modular-Stacker.
It performs Z-axis blending, applies the Z-LUT, and then
processes the image through the XY blending pipeline.
"""

import numpy as np
import cv2
from typing import List, Optional, Tuple, Any

# These imports are assumed to be other modules in the same project.
import weights
import lut_manager
import xy_blend_processor
from config import XYBlendOperation # Import XYBlendOperation to access its attributes

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance
    # Also set config reference for imported modules
    weights.set_config_reference(config_instance)
    lut_manager.set_config_reference(config_instance)
    xy_blend_processor.set_config_reference(config_instance)


def process_image_stack(image_window: List[np.ndarray]) -> np.ndarray:
    """
    Processes a single stack (window) of grayscale images.

    Args:
        image_window (List[np.ndarray]): A list of 8-bit grayscale NumPy arrays
                                         representing the image slices in the window.
                                         Can contain blank images (all zeros).

    Returns:
        np.ndarray: The final 8-bit grayscale NumPy array after stacking,
                    Z-LUT application, and XY pipeline processing.
    """
    if _config_ref is None:
        raise RuntimeError("Config reference not set in stacking_processor. Cannot process stack.")

    if not image_window:
        raise ValueError("Image window cannot be empty.")

    # Infer image shape from the first image in the window.
    # Assuming all images in the window are of the same shape.
    H, W = image_window[0].shape[:2] # Assuming grayscale (H, W) or (H, W, 1)

    # --- 1. Z-axis Blending ---
    blend_mode = _config_ref.blend_mode
    z_blended_image: Optional[np.ndarray] = None # Initialize to None

    if blend_mode in ["flat", "linear", "cosine", "exp_decay", "gaussian"]:
        # Convert images to 16-bit for accumulation to avoid overflow during weighted sum.
        # The `scale_bits` from config determines the precision for fixed-point arithmetic.
        scale_factor = 1 << _config_ref.scale_bits # e.g., 2^12 = 4096

        # Initialize accumulator for 32-bit to prevent overflow during intermediate sums
        accumulator = np.zeros((H, W), dtype=np.uint32)
        
        # Get blend mode and parameters from config
        blend_param = _config_ref.blend_param
        directional_blend = _config_ref.directional_blend
        dir_sigma = _config_ref.dir_sigma

        # Generate blending weights (float array)
        weights_float, _ = weights.generate_weights(
            length=len(image_window),
            blend_mode=blend_mode,
            blend_param=blend_param,
            directional=directional_blend,
            dir_sigma=dir_sigma
        )
        
        # Convert float weights to fixed-point integers.
        # These weights are scaled up by `scale_factor`.
        weights_int = [int(w * scale_factor) for w in weights_float]
        sum_weights_int = sum(weights_int) or 1 # Avoid division by zero

        for i, img_8bit in enumerate(image_window):
            # Accumulate the weighted sum.
            # `img_8bit` is a uint8 array (0-255). We cast it to uint32 to prevent overflow
            # during multiplication with `weights_int`, which can be large.
            accumulator += img_8bit.astype(np.uint32) * weights_int[i]

        # The accumulator now holds a value of `sum(image_i * weights_i_int)`.
        # To get the final normalized 8-bit value, we perform a rounding division.
        # `sum_weights_int` is approximately `scale_factor` if `sum(weights_float)` is 1.0.
        # This division effectively scales the result back down to the 0-255 range.
        # The `sum_weights_int // 2` term is for rounding (add half before integer division).
        blended_result_uint8 = ((accumulator + (sum_weights_int // 2)) // sum_weights_int).astype(np.uint8)
        
        z_blended_image = blended_result_uint8

    elif blend_mode in ["binary_contour", "gradient_contour"]:
        # These modes aggregate binary or gradient information.
        
        # We need to pass the config to these functions for thresholds etc.
        # Let's implement the logic directly here for these two modes.
        
        if blend_mode == "binary_contour":
            # Convert to binary (0 or 1)
            bin_stack = [(cv2.threshold(im, _config_ref.binary_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
            total_presence = np.sum(np.stack(bin_stack, axis=0), axis=0)
            # Normalize to 0-255
            norm = total_presence.astype(np.float32) / (total_presence.max() or 1)
            z_blended_image = (norm * 255).astype(np.uint8)
        elif blend_mode == "gradient_contour":
            # Convert to binary (0 or 1)
            bin_stack = [(cv2.threshold(im, _config_ref.gradient_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
            diffs = np.abs(np.diff(np.stack(bin_stack, axis=0), axis=0)) # Absolute difference between adjacent slices
            grad = np.sum(diffs, axis=0) # Sum of absolute differences (gradient magnitude)
            # Normalize to 0-255
            norm = grad.astype(np.float32) / (grad.max() or 1)
            z_blended_image = (norm * 255).astype(np.uint8)

    elif blend_mode in ["z_column_lift", "z_contour_interp"]:
        # These modes were previously in other files and have been adapted to work with the image_window.
        scale_factor = 1 << _config_ref.scale_bits
        
        if blend_mode == "z_column_lift":
            # Build fixed-point kernel using current blend_mode/param (from config)
            kernel_float, _ = weights.generate_weights(
                length=len(image_window),
                blend_mode=_config_ref.blend_mode, # Use the main blend mode for kernel shape
                blend_param=_config_ref.blend_param,
                directional=False, # Z-column lift typically isn't directional in this sense
                dir_sigma=0.0
            )
            kernel_int = np.round(np.array(kernel_float) * scale_factor).astype(np.uint16)
            
            # Use tensordot for an efficient weighted sum along the Z-axis.
            arr_uint8 = np.stack(image_window, axis=0) # (Z, H, W)
            accumulator = np.tensordot(kernel_int, arr_uint8.astype(np.uint32), axes=(0,0))

            # Normalize and clip to 0-255
            z_blended_image = (accumulator // scale_factor).clip(0, 255).astype(np.uint8)

            # Apply top surface smoothing if enabled
            if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
                z_blended_image = cv2.GaussianBlur(z_blended_image, (0,0), _config_ref.top_surface_strength)

        elif blend_mode == "z_contour_interp":
            # Load binary stack (0 or 1)
            stacks_binary = []
            for img_8bit in image_window:
                _, bw = cv2.threshold(img_8bit, 128, 1, cv2.THRESH_BINARY) # Use 128 as default threshold
                stacks_binary.append(bw)
            arr_binary = np.stack(stacks_binary, axis=0) # (Z, H, W)

            # Projections
            base = np.max(arr_binary, axis=0) * 255 # Max projection along Z, scaled to 0-255
            xz = np.max(arr_binary, axis=1).astype(np.uint8) # Max projection along Y
            yz = np.max(arr_binary, axis=2).astype(np.uint8) # Max projection along X

            # Sobel with int16
            sx = cv2.Sobel(xz, cv2.CV_16S, 1, 0, ksize=3)
            sy = cv2.Sobel(yz, cv2.CV_16S, 1, 0, ksize=3)
            
            mx = np.max(np.abs(sx), axis=0).astype(np.uint16)
            my = np.max(np.abs(sy), axis=0).astype(np.uint16)
            
            # Normalize 0-255
            mx = ((mx * 255) // (mx.max() or 1)).astype(np.uint8)
            my = ((my * 255) // (my.max() or 1)).astype(np.uint8)
            
            # The outer product of the 1D projected gradients is a valid way to create a 2D contour map.
            contour = np.outer(my, mx) 
            
            # Blend base image with contour using top_surface_strength as alpha
            mixed = cv2.addWeighted(base.astype(np.float32), 1.0 - _config_ref.top_surface_strength,
                                    contour.astype(np.float32), _config_ref.top_surface_strength, 0)
            mixed = np.clip(mixed, 0, 255).astype(np.uint8)

            if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
                mixed = cv2.GaussianBlur(mixed, (0,0), _config_ref.top_surface_strength)
            z_blended_image = mixed
    else:
        # Fallback for unknown blend modes
        print(f"Warning: Unknown blend mode '{blend_mode}'. Returning first image in window.")
        z_blended_image = image_window[0].copy()

    if z_blended_image is None:
        raise ValueError("Z-blending failed. `z_blended_image` is None.")

    # --- 2. Z-LUT Application ---
    # The Z-LUT is expected to handle all intensity remapping, including
    # clamping, preserving black, and floor/ceil effects.
    final_image_after_z_lut = lut_manager.apply_z_lut(z_blended_image)

    # --- 3. XY Blending Pipeline Application ---
    # This will apply all configured XY operations (blur, sharpen, resize, etc.)
    final_processed_image = xy_blend_processor.process_xy_pipeline(final_image_after_z_lut)

    return final_processed_image

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Stacking Processor Module Test ---")

    # Local helper for mock to ensure odd kernel sizes
    def _mock_ensure_odd_ksize(val: int) -> int:
        """Ensures a kernel size is a positive, odd integer."""
        if val <= 0:
            return 1
        return val if val % 2 != 0 else val + 1

    class MockXYBlendOperation:
        def __init__(self, op_type: str, **kwargs):
            self.type = op_type
            # Initialize all possible parameters with their default values from config.py's XYBlendOperation
            # This ensures the mock object has all expected attributes
            self.gaussian_ksize_x: int = kwargs.get("gaussian_ksize_x", 3)
            self.gaussian_ksize_y: int = kwargs.get("gaussian_ksize_y", 3)
            self.gaussian_sigma_x: float = kwargs.get("gaussian_sigma_x", 0.0)
            self.gaussian_sigma_y: float = kwargs.get("gaussian_sigma_y", 0.0)
            self.bilateral_d: int = kwargs.get("bilateral_d", 9)
            self.bilateral_sigma_color: float = kwargs.get("bilateral_sigma_color", 75.0)
            self.bilateral_sigma_space: float = kwargs.get("bilateral_sigma_space", 75.0)
            self.median_ksize: int = kwargs.get("median_ksize", 5)
            self.unsharp_amount: float = kwargs.get("unsharp_amount", 1.0)
            self.unsharp_threshold: int = kwargs.get("unsharp_threshold", 0)
            self.unsharp_blur_ksize: int = kwargs.get("unsharp_blur_ksize", 5)
            self.unsharp_blur_sigma: float = kwargs.get("unsharp_blur_sigma", 0.0)
            self.resize_width: Optional[int] = kwargs.get("resize_width", None)
            self.resize_height: Optional[int] = kwargs.get("resize_height", None)
            self.resample_mode: str = kwargs.get("resample_mode", "LANCZOS4")

        def to_dict(self): # Required for Config.to_dict
            # This method should return a dictionary that *includes* the 'params' key
            # as the actual Config.XYBlendOperation.to_dict() does.
            # This mock's purpose is to simulate the attributes, not necessarily the serialization.
            # However, for consistency with the Config, let's make it match.
            data = {"type": self.type}
            params = {}
            if self.type == "gaussian_blur":
                params = {
                    "gaussian_ksize_x": self.gaussian_ksize_x,
                    "gaussian_ksize_y": self.gaussian_ksize_y,
                    "gaussian_sigma_x": self.gaussian_sigma_x,
                    "gaussian_sigma_y": self.gaussian_sigma_y,
                }
            elif self.type == "bilateral_filter":
                params = {
                    "bilateral_d": self.bilateral_d,
                    "bilateral_sigma_color": self.bilateral_sigma_color,
                    "bilateral_sigma_space": self.bilateral_sigma_space,
                }
            elif self.type == "median_blur":
                params = {
                    "median_ksize": self.median_ksize,
                }
            elif self.type == "unsharp_mask":
                params = {
                    "unsharp_amount": self.unsharp_amount,
                    "unsharp_threshold": self.unsharp_threshold,
                    "unsharp_blur_ksize": self.unsharp_blur_ksize,
                    "unsharp_blur_sigma": self.unsharp_blur_sigma,
                }
            elif self.type == "resize":
                params = {
                    "resize_width": self.resize_width,
                    "resize_height": self.resize_height,
                    "resample_mode": self.resample_mode,
                }
            data["params"] = params
            return data

        def __post_init__(self): # Simulate post_init for validation
            if self.type in ["gaussian_blur", "median_blur", "unsharp_mask"]:
                self.gaussian_ksize_x = _mock_ensure_odd_ksize(self.gaussian_ksize_x)
                self.gaussian_ksize_y = _mock_ensure_odd_ksize(self.gaussian_ksize_y)
                self.median_ksize = _mock_ensure_odd_ksize(self.median_ksize)
                self.unsharp_blur_ksize = _mock_ensure_odd_ksize(self.unsharp_blur_ksize)


    class MockConfig:
        def __init__(self):
            self.primary = 3
            self.radius = 1
            self.blend_mode = "gaussian" # Test with gaussian blend
            self.blend_param = 1.0
            self.directional_blend = False
            self.dir_sigma = 1.0
            self.scale_bits = 12 # Default for integer mode
            self.binary_threshold = 128
            self.gradient_threshold = 128
            self.top_surface_smoothing = False
            self.top_surface_strength = 0.5
            self.gradient_smooth = False # Not used in this module directly now
            self.gradient_blend_strength = 0.0 # Not used in this module directly now

            # LUT settings for testing
            self.lut_source = "generated"
            self.lut_generation_type = "linear"
            self.gamma_value = 1.0
            self.linear_min_input = 0
            self.linear_max_output = 255
            self.s_curve_contrast = 0.5
            self.log_param = 10.0
            self.exp_param = 2.0
            self.sqrt_param = 1.0
            self.rodbard_param = 1.0
            self.fixed_lut_path = ""

            # XY pipeline for testing
            self.xy_blend_pipeline: List[MockXYBlendOperation] = []

        def add_xy_op(self, op_type: str, **kwargs):
            self.xy_blend_pipeline.append(MockXYBlendOperation(op_type, **kwargs))

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config for all sub-modules

    # Ensure LUT is updated based on mock config
    lut_manager.update_active_lut_from_config()

    # Create dummy image window (5 slices for primary=3, radius=1)
    # Center slice (index 2) is brightest, fades out.
    dummy_images: List[np.ndarray] = []
    for i in range(5):
        img = np.zeros((50, 50), dtype=np.uint8)
        intensity = int(255 * (1 - abs(i - 2) / 2)) # Max at center, min at edges
        cv2.circle(img, (25, 25), 10 + i * 2, intensity, -1)
        dummy_images.append(img)
    
    # Add a blank image at the start and end to test None handling for radius
    # This simulates a window that extends beyond the actual image stack.
    # The image_loader would provide these as np.zeros.
    dummy_images_with_blanks = [np.zeros((50,50), dtype=np.uint8)] + dummy_images + [np.zeros((50,50), dtype=np.uint8)]
    print(f"Dummy image window length: {len(dummy_images_with_blanks)}")

    # Test Case 1: Gaussian blend with linear LUT and a simple XY blur
    print("\n--- Test Case 1: Gaussian Blend + Linear LUT + XY Gaussian Blur ---")
    mock_config.blend_mode = "gaussian"
    mock_config.blend_param = 1.0
    mock_config.directional_blend = False
    mock_config.lut_generation_type = "linear"
    mock_config.linear_min_input = 0
    mock_config.linear_max_output = 255
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("gaussian_blur", gaussian_ksize_x=5, gaussian_ksize_y=5, gaussian_sigma_x=1.0, gaussian_sigma_y=1.0)
    
    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_1 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_gaussian_linear_xyblur.png", output_img_1)
    print(f"Output 1 shape: {output_img_1.shape}, dtype: {output_img_1.dtype}")
    print("Output saved as test_stack_gaussian_linear_xyblur.png")

    # Test Case 2: Binary Contour blend + Gamma LUT + XY Resize
    print("\n--- Test Case 2: Binary Contour + Gamma LUT + XY Resize ---")
    mock_config.blend_mode = "binary_contour"
    mock_config.binary_threshold = 100 # Adjust threshold for binary conversion
    mock_config.lut_generation_type = "gamma"
    mock_config.gamma_value = 0.5 # Brightening gamma
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("resize", resize_width=100, resize_height=100, resample_mode="BICUBIC")

    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_2 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_binary_gamma_xyresize.png", output_img_2)
    print(f"Output 2 shape: {output_img_2.shape}, dtype: {output_img_2.dtype}")
    print("Output saved as test_stack_binary_gamma_xyresize.png")

    # Test Case 3: Z-Column Lift + S-Curve LUT + XY Unsharp Mask
    print("\n--- Test Case 3: Z-Column Lift + S-Curve LUT + XY Unsharp Mask ---")
    mock_config.blend_mode = "z_column_lift"
    mock_config.top_surface_smoothing = True # Enable for Z-column lift
    mock_config.top_surface_strength = 0.8 # Stronger smoothing
    mock_config.lut_generation_type = "s_curve"
    mock_config.s_curve_contrast = 0.9
    mock_config.xy_blend_pipeline = []
    mock_config.add_xy_op("unsharp_mask", unsharp_amount=1.5, unsharp_threshold=10)

    lut_manager.update_active_lut_from_config() # Update LUT based on config
    output_img_3 = process_image_stack(dummy_images_with_blanks)
    cv2.imwrite("test_stack_zcolumnlift_scurve_xyunsharp.png", output_img_3)
    print(f"Output 3 shape: {output_img_3.shape}, dtype: {output_img_3.dtype}")
    print("Output saved as test_stack_zcolumnlift_scurve_xyunsharp.png")

    # Clean up test images
    import os
    for f in ["test_stack_gaussian_linear_xyblur.png", 
              "test_stack_binary_gamma_xyresize.png", 
              "test_stack_zcolumnlift_scurve_xyunsharp.png"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")
