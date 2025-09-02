import tiledb
import numpy as np
import cv2
import os
from typing import List

def ingest_images_to_tiledb(image_files: List[str], tiledb_uri: str, logger):
    """
    Ingests a list of PNG images into a TileDB 3D array.
    """
    if not image_files:
        logger.log("No image files provided for TileDB ingestion.")
        return

    # Get dimensions from the first image
    first_image = cv2.imread(image_files[0], cv2.IMREAD_GRAYSCALE)
    if first_image is None:
        raise ValueError(f"Could not read the first image: {image_files[0]}")
    height, width = first_image.shape
    num_images = len(image_files)

    # Create TileDB array schema
    y_tile = min(height, 1024)
    x_tile = min(width, 1024)

    dom = tiledb.Domain(
        tiledb.Dim(name="z", domain=(0, num_images - 1), tile=1, dtype=np.uint32),
        tiledb.Dim(name="y", domain=(0, height - 1), tile=y_tile, dtype=np.uint32),
        tiledb.Dim(name="x", domain=(0, width - 1), tile=x_tile, dtype=np.uint32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
    )

    # Create the TileDB array
    if tiledb.array_exists(tiledb_uri):
        logger.log(f"Removing existing TileDB array at: {tiledb_uri}")
        tiledb.remove(tiledb_uri)

    logger.log(f"Creating new TileDB array at: {tiledb_uri}")
    tiledb.Array.create(tiledb_uri, schema)

    # Write images to the array
    with tiledb.open(tiledb_uri, "w") as A:
        for i, image_file in enumerate(image_files):
            logger.log(f"Ingesting image {i+1}/{num_images}: {os.path.basename(image_file)}")
            img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.log(f"Warning: Could not load image at {image_file}, skipping.")
                continue
            if img.shape != (height, width):
                logger.log(f"Warning: Image {image_file} has different dimensions, skipping.")
                continue
            A[i, :, :] = img
    logger.log("TileDB ingestion complete.")

def get_slice_and_binary(tiledb_uri: str, index: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads a single 2D slice from the TileDB array and returns it along with a binary version.
    """
    with tiledb.open(tiledb_uri, "r") as A:
        original_image = A[index, :, :]["pixel_value"]

    if original_image is None:
        return None, None

    _, binary_img = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
    return binary_img, original_image

# --- Placeholder functions for future implementation ---

def get_xz_slice(tiledb_uri: str, y_index: int) -> np.ndarray:
    """
    (Not yet implemented) Reads a single XZ slice from the TileDB array.
    """
    with tiledb.open(tiledb_uri, 'r') as A:
        # Slicing along the 'y' dimension
        data = A[:, y_index, :]["pixel_value"]
    return data


def get_yz_slice(tiledb_uri: str, x_index: int) -> np.ndarray:
    """
    (Not yet implemented) Reads a single YZ slice from the TileDB array.
    """
    with tiledb.open(tiledb_uri, 'r') as A:
        # Slicing along the 'x' dimension
        data = A[:, :, x_index]["pixel_value"]
    return data
