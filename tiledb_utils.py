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

import tiledb
import numpy as np
import os
import cv2
import shutil
from typing import List, Tuple

def create_dense_array_for_slices(uri: str, height: int, width: int, num_layers: int):
    """
    Creates a 3D dense TileDB array to store image slices.
    This version is robust against pre-existing, invalid, or mismatched arrays.

    Args:
        uri (str): The URI for the TileDB array.
        height (int): The height of the images.
        width (int): The width of the images.
        num_layers (int): The number of layers (Z dimension).
    """
    if tiledb.object_type(uri) == "array":
        with tiledb.open(uri, 'r') as A:
            if A.schema.domain.shape == (num_layers, height, width):
                print(f"Array with matching shape already exists at '{uri}'. Re-using.")
                return
            else:
                print(f"Array with conflicting shape found at '{uri}'. Removing and recreating.")
                shutil.rmtree(uri)
    elif os.path.exists(uri):
        print(f"Non-array file/folder found at '{uri}'. Removing and recreating.")
        shutil.rmtree(uri)

    print(f"Creating new TileDB array at '{uri}' with shape ({num_layers}, {height}, {width})")

    # Use a dimensionally-optimized tile layout for balanced performance
    dom = tiledb.Domain(
        tiledb.Dim(name="Z", domain=(0, num_layers - 1), tile=min(16, num_layers), dtype=np.uint32),
        tiledb.Dim(name="Y", domain=(0, height - 1), tile=min(256, height), dtype=np.uint32),
        tiledb.Dim(name="X", domain=(0, width - 1), tile=min(256, width), dtype=np.uint32),
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
        cell_order='row-major',
        tile_order='row-major',
    )

    tiledb.Array.create(uri, schema)

def ingest_images_to_tiledb(uri: str, image_files: List[str], input_path: str, batch_size=32):
    """
    Ingests a list of image files into a TileDB array using batching for performance.

    Args:
        uri (str): The URI of the TileDB array.
        image_files (List[str]): A sorted list of image filenames.
        input_path (str): The path to the directory containing the images.
        batch_size (int): The number of images to read and write per chunk.
    """
    if not image_files:
        raise ValueError("Image file list cannot be empty.")

    # Get dimensions from the first image
    first_image_path = os.path.join(input_path, image_files[0])
    img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read the first image: {first_image_path}")
    height, width = img.shape
    num_layers = len(image_files)

    create_dense_array_for_slices(uri, height, width, num_layers)

    with tiledb.open(uri, 'w') as A:
        for i in range(0, num_layers, batch_size):
            batch_filenames = image_files[i:i+batch_size]
            actual_batch_size = len(batch_filenames)

            # Pre-allocate numpy array for the batch
            batch_data = np.zeros((actual_batch_size, height, width), dtype=np.uint8)

            # Read images into the batch array
            for j, filename in enumerate(batch_filenames):
                filepath = os.path.join(input_path, filename)
                img_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img_data is not None:
                    batch_data[j, :, :] = img_data

            # Write the entire batch to TileDB
            print(f"Writing batch of {actual_batch_size} images to slice {i}...")
            A[i:i+actual_batch_size, :, :] = batch_data

def read_xy_slice(uri: str, slice_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a single XY slice from the TileDB array and returns it in the
    same format as core.load_image (binary_image, original_image).

    Args:
        uri (str): The URI of the TileDB array.
        slice_index (int): The Z-index of the slice to read.

    Returns:
        A tuple containing the binary image and the original grayscale image.
    """
    with tiledb.open(uri, 'r') as A:
        original_image = A[slice_index, :, :]["pixel_value"]

    _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)

    return binary_image, original_image

def read_xz_slice(uri: str, row_index: int) -> np.ndarray:
    """
    Reads a virtual XZ slice from the TileDB array.

    Args:
        uri (str): The URI of the TileDB array.
        row_index (int): The Y-index of the slice to read.

    Returns:
        A 2D numpy array representing the XZ slice.
    """
    with tiledb.open(uri, 'r') as A:
        # Slicing is [Z, Y, X]
        xz_slice = A[:, row_index, :]["pixel_value"]
    return xz_slice

def read_yz_slice(uri: str, col_index: int) -> np.ndarray:
    """
    Reads a virtual YZ slice from the TileDB array.

    Args:
        uri (str): The URI of the TileDB array.
        col_index (int): The X-index of the slice to read.

    Returns:
        A 2D numpy array representing the YZ slice.
    """
    with tiledb.open(uri, 'r') as A:
        # Slicing is [Z, Y, X]
        yz_slice = A[:, :, col_index]["pixel_value"]
    return yz_slice
