# Agent Guidelines

This document provides guidelines for AI agents working on this codebase.

## Performance Optimizations

### Use Numba for Numerical Loops

When implementing or refactoring functions that involve performance-critical, loop-heavy numerical operations (e.g., iterating over pixels in an image), please use the Numba library to accelerate the code.

**Example:**

Decorate the Python function with `@numba.jit`. Numba will compile it to high-performance machine code. This is particularly effective on functions that contain explicit `for` loops over NumPy arrays.

```python
import numba
import numpy as np

@numba.jit(nopython=True, cache=True)
def process_image_pixels(image_array):
    # Create a new array to store the output
    output_array = np.zeros_like(image_array)

    # Numba will heavily optimize this loop
    for y in range(image_array.shape[0]):
        for x in range(image_array.shape[1]):
            # Example operation
            output_array[y, x] = image_array[y, x] * 2

    return output_array
```

The `_calculate_receding_gradient_field_enhanced_edt_numba` function in `processing_core.py` serves as a reference implementation of this pattern.
