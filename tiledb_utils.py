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

def create_sparse_array_for_slices(uri: str, height: int, width: int, num_layers: int):
    """
    Creates a 3D sparse TileDB array to store image slices.
    This version is robust against pre-existing, invalid, or mismatched arrays.
    """
    if tiledb.object_type(uri) == "array":
        with tiledb.open(uri, 'r') as A:
            if A.schema.sparse and A.schema.domain.shape == (num_layers, height, width):
                print(f"Sparse array with matching shape already exists at '{uri}'. Re-using.")
                return
            else:
                print(f"Array with conflicting schema (not sparse or wrong shape) found at '{uri}'. Removing and recreating.")
                shutil.rmtree(uri)
    elif os.path.exists(uri):
        print(f"Non-array file/folder found at '{uri}'. Removing and recreating.")
        shutil.rmtree(uri)

    print(f"Creating new SPARSE TileDB array at '{uri}' with shape ({num_layers}, {height}, {width})")

    dom = tiledb.Domain(
        tiledb.Dim(name="Z", domain=(0, num_layers - 1), tile=min(16, num_layers), dtype=np.uint32),
        tiledb.Dim(name="Y", domain=(0, height - 1), tile=min(256, height), dtype=np.uint32),
        tiledb.Dim(name="X", domain=(0, width - 1), tile=min(256, width), dtype=np.uint32),
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=True,
        attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
        cell_order='row-major',
        tile_order='row-major',
    )

    tiledb.Array.create(uri, schema)

def ingest_images_to_tiledb(uri: str, image_files: List[str], input_path: str, **kwargs):
    """
    Ingests a list of image files into a sparse TileDB array.
    This version processes one image at a time to keep memory usage low and constant,
    but performs all writes inside a single array opening to maintain performance.
    """
    if not image_files:
        raise ValueError("Image file list cannot be empty.")

    first_image_path = os.path.join(input_path, image_files[0])
    img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read the first image: {first_image_path}")
    height, width = img.shape
    num_layers = len(image_files)

    create_sparse_array_for_slices(uri, height, width, num_layers)

    print("Writing image data to sparse array (slice by slice)...")
    with tiledb.open(uri, 'w') as A:
        for i, filename in enumerate(image_files):
            filepath = os.path.join(input_path, filename)
            img_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

            if img_data is not None:
                non_empty_y, non_empty_x = np.where(img_data > 0)

                if non_empty_y.size > 0:
                    z_coords = np.full_like(non_empty_y, i)
                    data = img_data[non_empty_y, non_empty_x]

                    A[z_coords, non_empty_y, non_empty_x] = data

def _reconstruct_dense_slice(data: dict, shape: Tuple[int, int], dim_names: Tuple[str, str]) -> np.ndarray:
    """Helper to reconstruct a dense 2D slice from sparse query results."""
    dense_slice = np.zeros(shape, dtype=np.uint8)
    if not data or 'pixel_value' not in data or data['pixel_value'].size == 0:
        return dense_slice

    coords1 = data[dim_names[0]]
    coords2 = data[dim_names[1]]
    values = data['pixel_value']
    dense_slice[coords1, coords2] = values
    return dense_slice

def read_xy_slice(uri: str, slice_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads a single XY slice from the sparse TileDB array and reconstructs it.
    """
    with tiledb.open(uri, 'r') as A:
        height, width = A.schema.domain.dim("Y").shape[0], A.schema.domain.dim("X").shape[0]
        data = A.multi_index[slice_index]
        original_image = _reconstruct_dense_slice(data, (height, width), ("Y", "X"))

    _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
    return binary_image, original_image

def read_xz_slice(uri: str, row_index: int) -> np.ndarray:
    """
    Reads a virtual XZ slice from the sparse TileDB array and reconstructs it.
    """
    with tiledb.open(uri, 'r') as A:
        num_layers, width = A.schema.domain.dim("Z").shape[0], A.schema.domain.dim("X").shape[0]
        data = A.multi_index[:, row_index]
        xz_slice = _reconstruct_dense_slice(data, (num_layers, width), ("Z", "X"))
    return xz_slice

def read_yz_slice(uri: str, col_index: int) -> np.ndarray:
    """
    Reads a virtual YZ slice from the sparse TileDB array and reconstructs it.
    """
    with tiledb.open(uri, 'r') as A:
        num_layers, height = A.schema.domain.dim("Z").shape[0], A.schema.domain.dim("Y").shape[0]
        data = A.multi_index[:, :, col_index]
        yz_slice = _reconstruct_dense_slice(data, (num_layers, height), ("Z", "Y"))
    return yz_slice
