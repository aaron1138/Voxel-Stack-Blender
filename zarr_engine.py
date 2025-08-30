"""
Zarr Engine for Voxel Stack Processing

This module provides functionalities to handle Zarr storage for large voxel datasets,
including initialization of Zarr arrays and streaming image data into them.
"""

import os
import zarr
import numpy as np
import cv2
import concurrent.futures
from typing import List

def initialize_storage(path: str, layers: int, height: int, width: int, chunk_size: tuple):
    """
    Initializes a Zarr group with two datasets for original and modified voxels.

    Args:
        path (str): The directory path to store the Zarr data.
        layers (int): The number of layers (Z dimension).
        height (int): The height of each slice (Y dimension).
        width (int): The width of each slice (X dimension).
        chunk_size (tuple): The chunking strategy for the Zarr arrays.

    Returns:
        tuple: A tuple containing the Zarr arrays for original and modified voxels.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

    zarr_path = os.path.join(path, 'voxel_data.zarr')
    root = zarr.open(zarr_path, mode='w')

    shape = (layers, height, width)

    original_voxels = root.create_dataset(
        'original_voxels',
        shape=shape,
        chunks=chunk_size,
        dtype='uint8',
        overwrite=True
    )

    modified_voxels = root.create_dataset(
        'modified_voxels',
        shape=shape,
        chunks=chunk_size,
        dtype='uint8',
        overwrite=True
    )

    return original_voxels, modified_voxels

def _load_slice(args):
    """Helper function to load a single image slice for parallel processing."""
    filepath, zarr_array, index = args
    try:
        # Use load_image from processing_core to maintain consistency
        # For now, we assume a simplified loading, just reading the grayscale image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            zarr_array[index] = img
            return True
        else:
            print(f"Warning: Could not load image {filepath}")
            return False
    except Exception as e:
        print(f"Error loading slice {filepath}: {e}")
        return False

def load_slices_to_zarr(image_paths: List[str], zarr_array, max_workers: int):
    """
    Loads image slices from a list of file paths into a Zarr array in parallel.

    Args:
        image_paths (List[str]): A list of paths to the image files.
        zarr_array: The Zarr array to write the data into.
        max_workers (int): The number of parallel workers to use for loading.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Prepare arguments for each task
        tasks = [(path, zarr_array, i) for i, path in enumerate(image_paths)]

        # Use executor.map to process tasks in parallel
        results = list(executor.map(_load_slice, tasks))

        successful_loads = sum(1 for r in results if r)
        print(f"Successfully loaded {successful_loads}/{len(image_paths)} slices into Zarr.")
