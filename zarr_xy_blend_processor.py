import cv2
import numpy as np
import zarr
from typing import List
import lut_manager
from config import XYBlendOperation, Config

def apply_gaussian_blur(image: np.ndarray, op: XYBlendOperation, config: Config) -> np.ndarray:
    """Applies Gaussian blur, with optional anisotropic correction."""
    ksize_x = op.gaussian_ksize_x
    ksize_y = op.gaussian_ksize_y
    sigma_x = op.gaussian_sigma_x
    sigma_y = op.gaussian_sigma_y

    if op.anisotropic_correction_enabled:
        vx, vy = config.voxel_dim_x, config.voxel_dim_y
        if vx > 0 and vy > 0:
            aspect_ratio = vy / vx
            sigma_y = sigma_x * aspect_ratio

    return cv2.GaussianBlur(image, (ksize_x, ksize_y), sigmaX=sigma_x, sigmaY=sigma_y)

def apply_bilateral_filter(image: np.ndarray, op: XYBlendOperation, config: Config) -> np.ndarray:
    """Applies a bilateral filter, with optional anisotropic correction."""
    d = op.bilateral_d
    sigma_color = op.bilateral_sigma_color
    sigma_space = op.bilateral_sigma_space

    if op.anisotropic_correction_enabled:
        # OpenCV's bilateralFilter does not support anisotropic sigmaSpace.
        # As a workaround, we can scale sigmaSpace by the average anisotropy.
        vx, vy = config.voxel_dim_x, config.voxel_dim_y
        if vx > 0 and vy > 0:
            avg_anisotropy = (vx + vy) / (2 * vx) # Scale relative to X dimension
            sigma_space *= avg_anisotropy

    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Other functions from xy_blend_processor are copied here, but without the config param if not needed
def apply_median_blur(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    ksize = op.median_ksize
    if ksize <= 1: return image
    return cv2.medianBlur(image, ksize)

def apply_unsharp_mask(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    amount = op.unsharp_amount
    threshold = op.unsharp_threshold
    blur_ksize = op.unsharp_blur_ksize
    blur_sigma = op.unsharp_blur_sigma
    blurred_image = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=blur_sigma, sigmaY=blur_sigma)
    sharpened_image = cv2.addWeighted(image, 1.0 + amount, blurred_image, -amount, 0)
    if threshold > 0:
        diff = cv2.absdiff(image, sharpened_image)
        mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY_INV)[1]
        return np.where(mask == 255, image, sharpened_image)
    return sharpened_image

def apply_resize(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    width, height = op.resize_width, op.resize_height
    if width is None and height is None: return image
    current_height, current_width = image.shape[:2]
    if width is None: width = int(current_width * (height / current_height))
    elif height is None: height = int(current_height * (width / current_width))
    if width == current_width and height == current_height: return image
    flags = {"NEAREST": cv2.INTER_NEAREST, "BILINEAR": cv2.INTER_LINEAR, "BICUBIC": cv2.INTER_CUBIC, "LANCZOS4": cv2.INTER_LANCZOS4, "AREA": cv2.INTER_AREA}
    interp = flags.get(op.resample_mode.upper(), cv2.INTER_LANCZOS4)
    return cv2.resize(image, (width, height), interpolation=interp)

def apply_lut_operation(image: np.ndarray, op: XYBlendOperation) -> np.ndarray:
    # This function is simplified as it's not the focus of the Zarr task
    # It will just use the lut_manager as before.
    import os
    lut_params = op.lut_params
    generated_lut = None
    try:
        if lut_params.lut_source == "file" and lut_params.fixed_lut_path and os.path.exists(lut_params.fixed_lut_path):
            generated_lut = lut_manager.load_lut(lut_params.fixed_lut_path)
    except Exception as e:
        print(f"Error loading LUT: {e}")
    if generated_lut is None:
        generated_lut = np.arange(256, dtype=np.uint8) # Pass-through
    return cv2.LUT(image, generated_lut)


def process_xy_pipeline_zarr(input_stack: zarr.Array, config: Config) -> zarr.Array:
    """Applies the sequence of XY operations to each slice of a Zarr stack."""
    pipeline_ops = config.xy_blend_pipeline
    if not pipeline_ops or all(op.type == "none" for op in pipeline_ops):
        return input_stack

    output_stack = zarr.zeros_like(input_stack)

    op_map = {
        "gaussian_blur": apply_gaussian_blur,
        "bilateral_filter": apply_bilateral_filter,
        "median_blur": lambda img, op, cfg: apply_median_blur(img, op),
        "unsharp_mask": lambda img, op, cfg: apply_unsharp_mask(img, op),
        "resize": lambda img, op, cfg: apply_resize(img, op),
        "apply_lut": lambda img, op, cfg: apply_lut_operation(img, op),
    }

    for z in range(input_stack.shape[0]):
        processed_image = input_stack[z, :, :]
        for op in pipeline_ops:
            op_func = op_map.get(op.type)
            if op_func:
                if op.type in ["gaussian_blur", "bilateral_filter"]:
                    processed_image = op_func(processed_image, op, config)
                else:
                    processed_image = op_func(processed_image, op, config) # Pass config for consistency

            if processed_image.dtype != np.uint8:
                processed_image = np.clip(processed_image, 0, 255).astype(np.uint8)

        output_stack[z, :, :] = processed_image

    return output_stack
