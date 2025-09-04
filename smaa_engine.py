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

import cv2
import numpy as np
import numba

@numba.jit(nopython=True, cache=True)
def _find_and_blend_patterns_2x2(image: np.ndarray) -> np.ndarray:
    """
    Finds 2x2 aliasing patterns and applies grayscale blending.
    This is the core logic for a simplified morphological AA.

    Args:
        image: The original binary image (0 or 255).

    Returns:
        An 8-bit grayscale image with anti-aliased corners.
    """
    output_image = image.copy()
    height, width = image.shape

    # Iterate through each possible 2x2 block
    for y in range(height - 1):
        for x in range(width - 1):
            # Extract the 2x2 block
            p00 = image[y, x]
            p01 = image[y, x + 1]
            p10 = image[y + 1, x]
            p11 = image[y + 1, x + 1]

            # Sum the values (0 or 255). Using integer arithmetic avoids float issues.
            block_sum = int(p00) + int(p01) + int(p10) + int(p11)

            # Case 1: Three white pixels (255*3=765) and one black pixel (0)
            if block_sum == 765:
                if p00 == 0: output_image[y, x] = 128
                elif p01 == 0: output_image[y, x + 1] = 128
                elif p10 == 0: output_image[y + 1, x] = 128
                else: output_image[y + 1, x + 1] = 128

            # Case 2: Three black pixels (0) and one white pixel (255)
            elif block_sum == 255:
                if p00 == 255: output_image[y, x] = 128
                elif p01 == 255: output_image[y, x + 1] = 128
                elif p10 == 255: output_image[y + 1, x] = 128
                else: output_image[y + 1, x + 1] = 128

    return output_image


def apply_smaa(image: np.ndarray) -> np.ndarray:
    """
    Applies a simplified Morphological Anti-Aliasing effect to a binary image.
    This implementation uses a 2x2 pattern matching approach.

    Args:
        image: A binary (0 or 255) numpy array.

    Returns:
        An 8-bit grayscale image with anti-aliased edges.
    """
    if image is None or image.ndim != 2:
        raise ValueError("Input must be a 2D numpy array.")

    # Ensure the image is binary (0 or 255)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # The core logic is now in this numba-optimized function
    output_image = _find_and_blend_patterns_2x2(binary_image)

    return output_image
