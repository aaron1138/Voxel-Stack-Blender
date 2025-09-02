import numpy as np
import tiledb
import cv2
from scipy import ndimage
from config import Config

def blend_orthogonal(
    tiledb_array_uri: str,
    image_shape: tuple,
    current_z: int,
    z_blended_image: np.ndarray,
    config: Config
) -> np.ndarray:
    """
    Performs orthogonal blending on XZ and YZ planes for better anti-aliasing.

    This function identifies the edges of features in the Z-blended image,
    reads the corresponding XZ and YZ planes from the TileDB array, applies a
    1D Gaussian blur along the Z-axis of these planes, and then merges the
    resulting smoothed pixel values back into the original image.

    Args:
        tiledb_array_uri (str): The URI of the TileDB array containing the full image stack.
        image_shape (tuple): The shape of the full image stack (depth, height, width).
        current_z (int): The index of the current slice being processed.
        z_blended_image (np.ndarray): The image after initial Z-blending has been applied.
        config (Config): The application configuration object.

    Returns:
        np.ndarray: The image with orthogonal blending applied.
    """
    depth, height, width = image_shape
    output_image = z_blended_image.copy()
    sigma = config.tiledb_orthogonal_blur_sigma

    if sigma <= 0:
        return output_image

    # 1. Find edges in the current Z-blended slice to determine where to apply the effect.
    # We use a binary version of the image for clear edge detection.
    _, binary_image = cv2.threshold(z_blended_image, 1, 255, cv2.THRESH_BINARY)
    edges = cv2.Canny(binary_image, 100, 200)
    edge_coords = np.argwhere(edges > 0)

    if edge_coords.size == 0:
        return output_image # No edges to process

    # 2. Open the TileDB array to read orthogonal slices
    try:
        with tiledb.open(tiledb_array_uri, 'r') as A:
            # 3. Iterate over the edge pixels and apply orthogonal blending
            for y, x in edge_coords:
                # Read the full XZ and YZ planes for the current pixel coordinate
                xz_plane = A[:, y, :]
                yz_plane = A[:, :, x]

                # Apply 1D Gaussian blur along the Z-axis (axis=0)
                # We convert to float for the filter and back to uint8 later
                blurred_xz = ndimage.gaussian_filter1d(xz_plane.astype(float), sigma=sigma, axis=0)
                blurred_yz = ndimage.gaussian_filter1d(yz_plane.astype(float), sigma=sigma, axis=0)

                # Sample the new pixel value from the blurred planes at the current Z index
                new_val_from_xz = blurred_xz[current_z, x]
                new_val_from_yz = blurred_yz[current_z, y]

                # Choose the stronger of the two blending values
                merged_ortho_val = max(new_val_from_xz, new_val_from_yz)

                # Merge the orthogonal value with the existing Z-blended value
                # We use maximum to ensure we don't darken existing features
                current_pixel_val = output_image[y, x]
                output_image[y, x] = np.uint8(max(current_pixel_val, merged_ortho_val))

    except tiledb.TileDBError as e:
        print(f"Error during orthogonal blending: {e}")
        # Return the original z-blended image if TileDB access fails
        return z_blended_image

    return output_image
