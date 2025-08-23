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

# A cache for the generated table to avoid re-computation
_gradient_table_cache = {}

def generate_linear_table(size=256):
    """
    Generates a triangular 2D list for gradient lookups and caches it.
    Row 'd' (1-indexed) in the list will contain 'd' linearly spaced grayscale values.

    For a distance 'd', the gradient values are evenly spaced between white (255)
    and black (0). For example, for d=2, there are two intermediate values, which
    are at 2/3 and 1/3 of the grayscale range, resulting in [170, 85].

    Args:
        size (int): The maximum distance to generate gradients for (number of rows).

    Returns:
        list: A list of lists, where table[d] contains the gradient for distance d.
              The list is 1-indexed (index 0 is a placeholder).
    """
    if size in _gradient_table_cache:
        return _gradient_table_cache[size]

    # Index 0 is a placeholder since distances are 1-based.
    gradient_table = [[0]]

    for d in range(1, size + 1):
        # Generate d+2 values from 255 down to 0, e.g., for d=2 -> [255, 170, 85, 0].
        # Then, take the d intermediate values.
        values = np.linspace(255, 0, d + 2)[1:-1]
        gradient_table.append(np.round(values).astype(np.uint8).tolist())

    _gradient_table_cache[size] = gradient_table
    return gradient_table

# Pre-generate and cache the default table on module import
generate_linear_table()
