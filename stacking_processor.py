# stacking_processor.py

"""
The core image processing engine for Modular-Stacker.
This version has been refactored to clearly separate the three main processing modes:
1. Vertical Blending as a pre-processor, followed by Z-axis Stacking.
2. Vertical Blending as a substitute for Z-axis Stacking.
3. Standard Z-axis Stacking only.
"""

import numpy as np
import cv2
from typing import List, Optional, Any

import weights
import lut_manager
import xy_blend_processor
import vertical_blend_processor
from config import XYBlendOperation

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance
    # Also set config reference for imported modules
    weights.set_config_reference(config_instance)
    lut_manager.set_config_reference(config_instance)
    xy_blend_processor.set_config_reference(config_instance)
    vertical_blend_processor.set_config_reference(config_instance)


def process_image_stack(image_window: List[np.ndarray]) -> np.ndarray:
    """
    Processes a single stack (window) of grayscale images through the full pipeline,
    dispatching to the correct processing mode based on configuration.
    """
    if _config_ref is None:
        raise RuntimeError("Config reference not set in stacking_processor.")
    if not image_window:
        raise ValueError("Image window cannot be empty.")

    # --- Determine the processing path ---
    is_vb_substitute = (
        not _config_ref.vertical_blend_pre_process and 
        _config_ref.blend_mode.startswith("vertical_")
    )
    is_vb_preprocess = _config_ref.vertical_blend_pre_process

    # This variable will hold the single, blended image result from the Z-axis processing.
    stacked_image: np.ndarray

    # --- STAGE 1: Perform the primary Z-axis blend/stack operation ---

    if is_vb_substitute:
        # --- MODE 1: Vertical Blending ONLY (Substitute Mode) ---
        # This mode calls the new function that directly produces the final blended image.
        stacked_image = vertical_blend_processor.blend_image_window_to_single_image(image_window)
        
        # Apply vertical LUTs if configured (logic preserved from original)
        if _config_ref.apply_vertical_luts:
            stacked_image = lut_manager.apply_z_lut(stacked_image, "receding")
            stacked_image = lut_manager.apply_z_lut(stacked_image, "overhang")

    elif is_vb_preprocess:
        # --- MODE 2: Sequential (VB Pre-process -> Stacking) ---
        # First, run the vertical blend pre-processor on the window
        vb_processed_window = vertical_blend_processor.process_image_window(image_window)
        
        # Optionally apply vertical LUTs to the pre-processed window (logic preserved from original)
        if _config_ref.apply_vertical_luts:
            receded_luts_applied = [lut_manager.apply_z_lut(img, "receding") for img in vb_processed_window]
            vb_processed_window = [lut_manager.apply_z_lut(img, "overhang") for img in receded_luts_applied]
            
        # Then, perform the standard stacking on the result of the pre-processing
        stacked_image = _perform_stacking(vb_processed_window, _config_ref.blend_mode)
        
    else:
        # --- MODE 3: Stacking ONLY ---
        # Perform the standard stacking on the original, unmodified window
        stacked_image = _perform_stacking(image_window, _config_ref.blend_mode)

    # --- STAGE 2: Optional Default LUT Application ---
    # This LUT is applied after the Z-axis processing is complete.
    if _config_ref.apply_default_lut_after_stacking:
        image_ready_for_xy = lut_manager.apply_z_lut(stacked_image, "default")
    else:
        image_ready_for_xy = stacked_image

    # --- STAGE 3: XY Blending Pipeline ---
    # The final stage applies any configured XY-plane filters.
    final_processed_image = xy_blend_processor.process_xy_pipeline(image_ready_for_xy)

    return final_processed_image


def _perform_stacking(image_window: List[np.ndarray], blend_mode: str) -> np.ndarray:
    """
    Performs the standard Z-axis stacking operation on a window of images.
    This function contains the original stacking logic and is unchanged.
    """
    H, W = image_window[0].shape[:2]
    scale_factor = 1 << _config_ref.scale_bits

    if blend_mode in ["flat", "linear", "cosine", "exp_decay", "gaussian"]:
        accumulator = np.zeros((H, W), dtype=np.uint32)
        weights_float, _ = weights.generate_weights(
            length=len(image_window),
            blend_mode=blend_mode,
            blend_param=_config_ref.blend_param,
            directional=_config_ref.directional_blend,
            dir_sigma=_config_ref.dir_sigma
        )
        weights_int = [int(w * scale_factor) for w in weights_float]
        sum_weights_int = sum(weights_int) or 1

        for i, img_8bit in enumerate(image_window):
            accumulator += img_8bit.astype(np.uint32) * weights_int[i]
        
        return ((accumulator + (sum_weights_int // 2)) // sum_weights_int).astype(np.uint8)

    elif blend_mode == "binary_contour":
        bin_stack = [(cv2.threshold(im, _config_ref.binary_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
        total_presence = np.sum(np.stack(bin_stack, axis=0), axis=0)
        norm = total_presence.astype(np.float32) / (total_presence.max() or 1)
        return (norm * 255).astype(np.uint8)

    elif blend_mode == "gradient_contour":
        bin_stack = [(cv2.threshold(im, _config_ref.gradient_threshold, 255, cv2.THRESH_BINARY)[1] // 255) for im in image_window]
        diffs = np.abs(np.diff(np.stack(bin_stack, axis=0), axis=0))
        grad = np.sum(diffs, axis=0)
        norm = grad.astype(np.float32) / (grad.max() or 1)
        return (norm * 255).astype(np.uint8)
    
    elif blend_mode == "z_column_lift":
        arr_uint8 = np.stack(image_window, axis=0)
        kernel_float, _ = weights.generate_weights(
            length=len(image_window), blend_mode=_config_ref.blend_mode,
            blend_param=_config_ref.blend_param, directional=False, dir_sigma=0.0
        )
        kernel_int = np.round(np.array(kernel_float) * scale_factor).astype(np.uint16)
        accumulator = np.tensordot(kernel_int, arr_uint8.astype(np.uint32), axes=(0,0))
        z_blended_image = (accumulator // scale_factor).clip(0, 255).astype(np.uint8)
        if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
            z_blended_image = cv2.GaussianBlur(z_blended_image, (0,0), _config_ref.top_surface_strength)
        return z_blended_image

    elif blend_mode == "z_contour_interp":
        stacks_binary = [cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)[1] for img in image_window]
        arr_binary = np.stack(stacks_binary, axis=0)
        base = np.max(arr_binary, axis=0) * 255
        xz = np.max(arr_binary, axis=1).astype(np.uint8)
        yz = np.max(arr_binary, axis=2).astype(np.uint8)
        sx = cv2.Sobel(xz, cv2.CV_16S, 1, 0, ksize=3)
        sy = cv2.Sobel(yz, cv2.CV_16S, 1, 0, ksize=3)
        mx = np.max(np.abs(sx), axis=0).astype(np.uint16)
        my = np.max(np.abs(sy), axis=0).astype(np.uint16)
        mx = ((mx * 255) // (mx.max() or 1)).astype(np.uint8)
        my = ((my * 255) // (my.max() or 1)).astype(np.uint8)
        contour = np.outer(my, mx)
        mixed = cv2.addWeighted(base.astype(np.uint8), 1.0 - _config_ref.top_surface_strength,
                                contour, _config_ref.top_surface_strength, 0)
        if _config_ref.top_surface_smoothing and _config_ref.top_surface_strength > 0:
            mixed = cv2.GaussianBlur(mixed, (0,0), _config_ref.top_surface_strength)
        return mixed

    else:
        # Default fallback: return the central image of the window
        center_index = len(image_window) // 2
        return image_window[center_index].copy()
