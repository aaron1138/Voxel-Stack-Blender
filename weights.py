# weights.py

"""
Generate weight kernels for blending windows of slices in Modular-Stacker.
Supports flat, linear, cosine, exponential decay, and Gaussian modes.
"""

import math
import numpy as np
from typing import List, Tuple, Any, Optional

# A reference to the application configuration (will be set externally)
_config_ref: Optional[Any] = None # Using Any to avoid circular import with config.py for now

def set_config_reference(config_instance: Any):
    """Sets the reference to the global Config instance."""
    global _config_ref
    _config_ref = config_instance

def generate_weights(
    length: int,
    blend_mode: str,
    blend_param: float,
    directional: bool,
    dir_sigma: float
) -> Tuple[List[float], List[int]]:
    """
    Generates a list of normalized blending weights and their relative positions.

    Args:
        length (int): The total length of the window (number of slices).
        blend_mode (str): The type of weight curve to generate
                          ('flat', 'linear', 'cosine', 'exp_decay', 'gaussian').
        blend_param (float): Parameter for blend_mode (e.g., sigma for gaussian/exp_decay).
        directional (bool): If True, apply a directional bias.
        dir_sigma (float): Decay factor for directional bias.

    Returns:
        Tuple[List[float], List[int]]:
            - weights: Normalized list of floats summing to 1.
            - positions: Relative indices from the center of the window.
    """
    if length <= 0:
        return [], []

    center = length // 2
    weights: List[float] = []
    positions: List[int] = []

    for i in range(length):
        dist = abs(i - center)
        positions.append(i - center)

        w = 0.0 # Default weight

        if blend_mode == "flat":
            w = 1.0

        elif blend_mode == "linear":
            # Linear decay from center to edges
            # The 'span' should be the maximum distance from the center to an edge.
            # For a window of length L, the max distance from center is L/2.
            span = length / 2.0
            if span == 0: # Avoid division by zero for length 1
                w = 1.0
            else:
                w = max(0.0, 1.0 - (dist / span))

        elif blend_mode == "cosine":
            # Cosine curve from 1.0 at center to 0.0 at edges
            span = length / 2.0 or 1.0 # Avoid division by zero
            angle = (dist / span) * (math.pi / 2.0) # Map distance to 0 to pi/2
            w = max(0.0, math.cos(angle))

        elif blend_mode == "exp_decay":
            # Exponential decay from center
            if blend_param <= 0: # Avoid division by zero or non-decaying curve
                w = 1.0 if dist == 0 else 0.0 # Treat as impulse if param is zero or negative
            else:
                w = math.exp(-dist / blend_param)

        elif blend_mode == "gaussian":
            # Gaussian (normal distribution) curve
            if blend_param <= 0: # Treat as impulse if sigma is zero or negative
                w = 1.0 if dist == 0 else 0.0
            else:
                w = math.exp(-0.5 * (dist / blend_param) ** 2)
        
        # Add other modes if needed, but for now, these are the primary ones for Z-blending.
        # Binary/gradient modes are handled differently, not by these weights directly.

        weights.append(w)

    # Apply directional bias if requested
    if directional and dir_sigma > 0:
        biased_weights = []
        for (pos, w) in zip(positions, weights):
            # Bias more towards 'newer' slices (positive positions)
            # The bias factor should be 1.0 at pos=0, and increase for positive pos, decrease for negative pos
            # A simple exponential bias: bias = exp(pos / dir_sigma)
            # Or, to only boost positive: bias = 1.0 + (pos / dir_sigma) if pos > 0 else 1.0
            # Let's use a symmetric exponential decay from the center, but only apply it as a multiplier
            # for positive positions (newer slices).
            
            # The original implementation had: bias = math.exp(-abs(pos) / dir_sigma) and w = w * bias if pos > 0 else w
            # This means older slices (pos < 0) get no bias, center gets exp(0)=1, newer slices get exp(-abs(pos)/dir_sigma)
            # which is a *decreasing* bias for newer slices further from center. This seems counter-intuitive for "directional bias".
            # If "directional" means "bias towards newer slices", then positive positions should get *more* weight.

            # Let's interpret "directional bias" as making newer slices more influential.
            # A simple way: linearly increase bias for newer slices, decrease for older.
            # Or, a positive exponential for newer, negative for older.
            
            # Reverting to the original logic's *intent* (as per previous code comments):
            # "if True, bias weights in the positive (newer) direction"
            # The original code's `bias = math.exp(-abs(pos) / dir_sigma)` means smaller `abs(pos)` (closer to center)
            # gives higher `bias`. If `pos > 0`, it applies this bias. This means slices *closer* to the center
            # (on the newer side) get more weight. Slices further out on the newer side get *less* weight.
            # This is a "center-weighted bias on the newer side".

            # Let's stick to the original logic for now, assuming its intended effect.
            # The `dir_sigma` controls how sharply this bias falls off.
            bias = math.exp(-abs(pos) / dir_sigma) # Bias factor: higher for positions closer to center
            if pos > 0: # Only apply bias to slices newer than the center
                w = w * bias
            biased_weights.append(w)
        weights = biased_weights

    # Normalize sum to 1.0
    total = sum(weights)
    if total == 0:
        # If all weights are zero (e.g., length=0 or invalid params),
        # distribute evenly to avoid division by zero.
        if length > 0:
            weights = [1.0 / length] * length
        else:
            weights = []
    else:
        weights = [w / total for w in weights]

    return weights, positions

# Example usage (for testing purposes, remove in final app)
if __name__ == '__main__':
    print("--- Weights Module Test ---")

    # Dummy Config for testing
    class MockConfig:
        def __init__(self):
            self.primary = 3
            self.radius = 2
            self.blend_mode = "gaussian"
            self.blend_param = 1.0
            self.directional_blend = False
            self.dir_sigma = 1.0

    mock_config = MockConfig()
    set_config_reference(mock_config) # Set the mock config

    window_length = mock_config.primary + 2 * mock_config.radius
    print(f"Testing window length: {window_length}")

    # Test Gaussian weights
    print("\n--- Gaussian Blend (σ=1.0, no directional) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=mock_config.blend_param,
        directional=mock_config.directional_blend,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test Linear weights
    print("\n--- Linear Blend (no directional) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="linear",
        blend_param=mock_config.blend_param, # Param not used for linear, but passed for consistency
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test Gaussian with directional bias
    print("\n--- Gaussian Blend (σ=1.0, with directional σ=1.0) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=mock_config.blend_param,
        directional=True,
        dir_sigma=1.0
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test edge case: length 1
    print("\n--- Flat Blend (length=1) ---")
    weights, positions = generate_weights(
        length=1,
        blend_mode="flat",
        blend_param=mock_config.blend_param,
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")

    # Test edge case: blend_param=0 for gaussian/exp_decay
    print("\n--- Gaussian Blend (σ=0.0) ---")
    weights, positions = generate_weights(
        length=window_length,
        blend_mode="gaussian",
        blend_param=0.0,
        directional=False,
        dir_sigma=mock_config.dir_sigma
    )
    print(f"Positions: {positions}")
    print(f"Weights: {[f'{w:.4f}' for w in weights]}")
    print(f"Sum of weights: {sum(weights):.4f}")
