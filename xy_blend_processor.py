# xy_blend_processor.py

"""
Processes an image through a sequence of XY blending, smoothing, sharpening,
and resizing operations as defined by the Config's xy_blend_pipeline.
"""

import cv2
import numpy as np
from typing import List, Any, Optional, Dict # Import Dict here

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def _ensure_odd_ksize(ksize: int) -> int:
    """Ensures a kernel size is an odd integer, adjusting if necessary."""
    if ksize % 2 == 0:
        return ksize + 1 if ksize > 0 else 1
    return ksize

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_gaussian_blur(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies Gaussian blur to an 8-bit grayscale image.
    Uses configurable separable X and Y kernel sizes and sigmas.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize_x = _ensure_odd_ksize(getattr(op, "gaussian_ksize_x", 3))
    ksize_y = _ensure_odd_ksize(getattr(op, "gaussian_ksize_y", 3))
    sigma_x = getattr(op, "gaussian_sigma_x", 0.0)
    sigma_y = getattr(op, "gaussian_sigma_y", 0.0)

    # Ensure at least a minimal kernel if both ksize and sigma are zero, to prevent error or no-op
    if ksize_x == 1 and ksize_y == 1 and sigma_x == 0.0 and sigma_y == 0.0:
        ksize_x, ksize_y = 3, 3 # Default to a small blur if no parameters given

    # cv2.GaussianBlur expects ksize as a tuple (width, height)
    return cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_bilateral_filter(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies a bilateral filter to an 8-bit grayscale image.
    Effective for noise reduction while preserving edges.
    """
    # Access attributes directly from the XYBlendOperation object
    d = getattr(op, "bilateral_d", 9)
    sigma_color = getattr(op, "bilateral_sigma_color", 75.0)
    sigma_space = getattr(op, "bilateral_sigma_space", 75.0)

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_median_blur(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies a median blur to an 8-bit grayscale image.
    Excellent for removing salt-and-pepper noise.
    """
    # Access attributes directly from the XYBlendOperation object
    ksize = _ensure_odd_ksize(getattr(op, "median_ksize", 5))
    if ksize <= 1: # Median blur requires kernel size > 1
        return image # No-op if kernel size is 1 or invalid
    return cv2.medianBlur(image, ksize)

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_unsharp_mask(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Applies unsharp masking to an 8-bit grayscale image for sharpening.
    """
    # Access attributes directly from the XYBlendOperation object
    amount = getattr(op, "unsharp_amount", 1.0)
    threshold = getattr(op, "unsharp_threshold", 0)
    blur_ksize = _ensure_odd_ksize(getattr(op, "unsharp_blur_ksize", 5))
    blur_sigma = getattr(op, "unsharp_blur_sigma", 0.0)

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

# Modified to accept XYBlendOperation directly instead of Dict[str, Any]
def apply_resize(image: np.ndarray, op: Any) -> np.ndarray:
    """
    Resizes an image to the specified width and height using the given resampling method.
    """
    # Access attributes directly from the XYBlendOperation object
    width = getattr(op, "resize_width", None)
    height = getattr(op, "resize_height", None)
    resample_mode = getattr(op, "resample_mode", "LANCZOS4").upper()

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


def process_xy_pipeline(image: np.ndarray) -> np.ndarray:
    """
    Applies the sequence of XY blending/processing operations defined in the Config.
    The input image is expected to be an 8-bit grayscale NumPy array.
    """
    if _config_ref is None:
        print("Warning: Config reference not set in xy_blend_processor. Cannot process XY pipeline.")
        return image

    processed_image = image.copy()

    for op in _config_ref.xy_blend_pipeline:
        op_type = op.type # Access the type directly from the XYBlendOperation object
        # op_params = op.params # This line caused the AttributeError, remove it

        if op_type == "none":
            continue # Skip no-op
        elif op_type == "gaussian_blur":
            processed_image = apply_gaussian_blur(processed_image, op) # Pass the op object directly
        elif op_type == "bilateral_filter":
            processed_image = apply_bilateral_filter(processed_image, op) # Pass the op object directly
        elif op_type == "median_blur":
            processed_image = apply_median_blur(processed_image, op) # Pass the op object directly
        elif op_type == "unsharp_mask":
            processed_image = apply_unsharp_mask(processed_image, op) # Pass the op object directly
        elif op_type == "resize":
            processed_image = apply_resize(processed_image, op) # Pass the op object directly
        else:
            print(f"Warning: Unknown XY blend operation type '{op_type}'. Skipping.")
    
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

    # Dummy Config and XYBlendOperation for testing the pipeline
    from config import Config, XYBlendOperation # Import actual classes for testing

    class MockConfig(Config): # Inherit from actual Config to get all fields
        def __init__(self):
            # Manually initialize _initialized to True to prevent recursive load from super().__init__()
            object.__setattr__(self, '_initialized', True)
            super().__init__() # Call parent dataclass init to set defaults
            self.xy_blend_pipeline: List[XYBlendOperation] = [] # Override with an empty list initially

        def add_op(self, op: XYBlendOperation):
            self.xy_blend_pipeline.append(op)

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    # Test pipeline: Gaussian Blur -> Resize -> Unsharp Mask
    print("\nTesting pipeline: Gaussian Blur -> Resize -> Unsharp Mask")
    mock_config.add_op(XYBlendOperation("gaussian_blur", gaussian_ksize_x=5, gaussian_ksize_y=5, gaussian_sigma_x=1.0, gaussian_sigma_y=1.0))
    mock_config.add_op(XYBlendOperation("resize", resize_width=50, resize_height=50, resample_mode="BICUBIC"))
    mock_config.add_op(XYBlendOperation("unsharp_mask", unsharp_amount=1.5, unsharp_threshold=10, unsharp_blur_ksize=3))

    output_image_pipeline = process_xy_pipeline(test_image)
    cv2.imwrite("test_xy_pipeline_output.png", output_image_pipeline)
    print(f"Pipeline output shape: {output_image_pipeline.shape}, dtype: {output_image_pipeline.dtype}")
    print("Pipeline output saved as test_xy_pipeline_output.png")

    # Test individual operations
    print("\nTesting individual operations:")

    # Gaussian Blur
    # Create a dummy XYBlendOperation for individual testing
    gaussian_op = XYBlendOperation(type="gaussian_blur", gaussian_ksize_x=7, gaussian_ksize_y=7, gaussian_sigma_x=1.5, gaussian_sigma_y=1.5)
    output_gaussian = apply_gaussian_blur(test_image, gaussian_op)
    cv2.imwrite("test_xy_gaussian.png", output_gaussian)
    print("Gaussian blur output saved as test_xy_gaussian.png")

    # Bilateral Filter
    bilateral_op = XYBlendOperation(type="bilateral_filter", bilateral_d=15, bilateral_sigma_color=80.0, bilateral_sigma_space=80.0)
    output_bilateral = apply_bilateral_filter(test_image, bilateral_op)
    cv2.imwrite("test_xy_bilateral.png", output_bilateral)
    print("Bilateral filter output saved as test_xy_bilateral.png")

    # Median Blur
    median_op = XYBlendOperation(type="median_blur", median_ksize=7)
    output_median = apply_median_blur(test_image, median_op)
    cv2.imwrite("test_xy_median.png", output_median)
    print("Median blur output saved as test_xy_median.png")

    # Unsharp Mask
    unsharp_op = XYBlendOperation(type="unsharp_mask", unsharp_amount=1.2, unsharp_threshold=5, unsharp_blur_ksize=5, unsharp_blur_sigma=0.0)
    output_unsharp = apply_unsharp_mask(test_image, unsharp_op)
    cv2.imwrite("test_xy_unsharp.png", output_unsharp)
    print("Unsharp mask output saved as test_xy_unsharp.png")

    # Resize
    resize_op = XYBlendOperation(type="resize", resize_width=75, resize_height=75, resample_mode="LANCZOS4")
    output_resize = apply_resize(test_image, resize_op)
    cv2.imwrite("test_xy_resize.png", output_resize)
    print("Resize output saved as test_xy_resize.png")

    # Clean up test images
    import os
    for f in ["test_xy_original.png", "test_xy_pipeline_output.png", 
              "test_xy_gaussian.png", "test_xy_bilateral.png", 
              "test_xy_median.png", "test_xy_unsharp.png", "test_xy_resize.png"]:
        if os.path.exists(f):
            os.remove(f)
            print(f"Cleaned up {f}")
