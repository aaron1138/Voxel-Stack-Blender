import tiledb
import numpy as np
import os
import cv2
from typing import List

def ingest_images_to_tiledb(image_files: List[str], array_uri: str, batch_size: int = 10):
    """
    Ingests a list of image files into a TileDB array.

    Args:
        image_files (List[str]): A sorted list of paths to the image files.
        array_uri (str): The URI for the new TileDB array.
        batch_size (int): The number of images to write in each batch.
    """
    if not image_files:
        raise ValueError("Image file list is empty.")

    # Remove the array if it already exists
    if tiledb.object_type(array_uri) == "array":
        tiledb.remove(array_uri)

    # Get dimensions from the first image
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise ValueError(f"Could not read the first image: {image_files[0]}")
    height, width = first_image.shape
    num_images = len(image_files)

    # Create the TileDB array schema
    dom = tiledb.Domain(
        tiledb.Dim(name="z", domain=(0, num_images - 1), tile=1, dtype=np.uint32),
        tiledb.Dim(name="y", domain=(0, height - 1), tile=256, dtype=np.uint32),
        tiledb.Dim(name="x", domain=(0, width - 1), tile=256, dtype=np.uint32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
        cell_order='row-major',
        tile_order='row-major'
    )
    tiledb.Array.create(array_uri, schema)

    # Write images to the array in batches
    with tiledb.open(array_uri, 'w') as A:
        for i in range(0, num_images, batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in batch_files]

            # Check for images that failed to load
            if any(img is None for img in batch_images):
                failed_files = [f for f, img in zip(batch_files, batch_images) if img is None]
                raise IOError(f"Failed to read one or more images: {', '.join(failed_files)}")

            # Check for consistent dimensions
            for j, img in enumerate(batch_images):
                if img.shape != (height, width):
                    raise ValueError(f"Image {batch_files[j]} has dimensions {img.shape}, but expected {(height, width)}")

            data = np.stack(batch_images)
            A[i:i+len(batch_files), :, :] = data
            print(f"Wrote layers {i} to {i+len(batch_files)-1} to TileDB array.")

def read_xy_slice(array_uri: str, z_index: int) -> np.ndarray:
    """Reads an XY slice from the TileDB array."""
    with tiledb.open(array_uri, 'r') as A:
        return A[z_index, :, :]["pixel_value"]

def read_xz_slice(array_uri: str, y_index: int) -> np.ndarray:
    """Reads an XZ slice from the TileDB array."""
    with tiledb.open(array_uri, 'r') as A:
        return A[:, y_index, :]["pixel_value"]

def read_yz_slice(array_uri: str, x_index: int) -> np.ndarray:
    """Reads a YZ slice from the TileDB array."""
    with tiledb.open(array_uri, 'r') as A:
        return A[:, :, x_index]["pixel_value"]
