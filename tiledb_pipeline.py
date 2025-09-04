import os
import tiledb
import numpy as np
import cv2
from scipy import ndimage
from config import Config
from roi_tracker import ROITracker
import processing_core as core
import re
import tempfile

def create_tiledb_array(array_path, height, width, num_layers):
    """Creates a 3D TileDB array to store the image stack."""
    if tiledb.array_exists(array_path):
        tiledb.remove(array_path)

    dom = tiledb.Domain(
        tiledb.Dim(name="Z", domain=(0, num_layers - 1), tile=1, dtype=np.uint32),
        tiledb.Dim(name="Y", domain=(0, height - 1), tile=height, dtype=np.uint32),
        tiledb.Dim(name="X", domain=(0, width - 1), tile=width, dtype=np.uint32),
    )

    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[
            tiledb.Attr(name="original", dtype=np.uint8),
            tiledb.Attr(name="binary", dtype=np.uint8),
        ],
    )

    tiledb.Array.create(array_path, schema)

def ingest_images_to_tiledb(array_path, image_files, start_index, stop_index):
    """Reads PNG images and writes them as slices to the TileDB array."""
    with tiledb.open(array_path, 'w') as A:
        for i, filepath in enumerate(image_files):
            if start_index is not None and i < start_index:
                continue
            if stop_index is not None and i > stop_index:
                continue

            binary_image, original_image = core.load_image(filepath)
            if original_image is not None:
                A[i, :, :] = {"original": original_image, "binary": binary_image}

def _calculate_3d_gradient_field(current_white_mask, prior_masks, config):
    """
    Calculates a 3D distance transform to create a smooth gradient field.
    """
    if not prior_masks:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # Create a 3D volume of the prior masks
    prior_volume = np.stack(prior_masks, axis=0)

    # Combine prior masks into a single 3D mask
    prior_white_combined_mask = np.any(prior_volume, axis=0).astype(np.uint8) * 255

    # Identify receding white areas
    receding_white_areas = cv2.bitwise_and(prior_white_combined_mask, cv2.bitwise_not(current_white_mask))
    if cv2.countNonZero(receding_white_areas) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # Create a 3D volume for the distance transform source
    # We are looking for the distance from the "solid" part of the current layer.
    distance_transform_src_3d = np.ones_like(prior_volume, dtype=np.uint8)
    # Set the last slice (which corresponds to the current layer) to be the source of the distance
    distance_transform_src_3d[-1] = cv2.bitwise_not(current_white_mask)

    # Perform 3D Euclidean Distance Transform
    # This calculates the distance from every "1" pixel to the nearest "0" pixel.
    distance_map_3d = ndimage.distance_transform_edt(distance_transform_src_3d)

    # We only care about the distance map for the final slice (the one we are generating the gradient for)
    distance_map_2d = distance_map_3d[-1]

    # Mask the distance map to only include the receding areas
    receding_distance_map = cv2.bitwise_and(distance_map_2d, distance_map_2d, mask=receding_white_areas)
    if np.max(receding_distance_map) == 0:
        return np.zeros_like(current_white_mask, dtype=np.uint8)

    # Normalize the gradient
    fade_dist = config.fixed_fade_distance_receding
    clipped_distance_map = np.clip(receding_distance_map, 0, fade_dist)
    denominator = fade_dist if fade_dist > 0 else 1.0
    normalized_map = (clipped_distance_map / denominator)

    inverted_normalized_map = 1.0 - normalized_map
    final_gradient_map = (255 * inverted_normalized_map).astype(np.uint8)

    # Final mask
    final_gradient_map = cv2.bitwise_and(final_gradient_map, final_gradient_map, mask=receding_white_areas)
    return final_gradient_map

def process_and_write_results(input_array_path, output_array_path, config: Config, image_filenames: list, get_numeric_part_func):
    """
    Processes Z-blending using data from the TileDB array and writes to an output array.
    """
    tracker = ROITracker()
    with tiledb.open(input_array_path, 'r') as A_in:
        num_layers, height, width = A_in.schema.domain.shape

        if tiledb.array_exists(output_array_path):
            tiledb.remove(output_array_path)

        create_output_array(output_array_path, height, width, num_layers)

        with tiledb.open(output_array_path, 'w') as A_out:
            for z, filename in enumerate(image_filenames):
                if z < config.receding_layers:
                    # Not enough prior layers, just write the original image
                    A_out[z, :, :] = A_in[z, :, :]["original"]
                    continue

                current_white_mask = A_in[z, :, :]["binary"]
                original_current_image = A_in[z, :, :]["original"]

                start_z = max(0, z - config.receding_layers)
                prior_masks_data = A_in[start_z:z, :, :]["binary"]

                prior_masks = [prior_masks_data[i, :, :] for i in range(prior_masks_data.shape[0])]

                if config.blending_mode == config.blending_mode.ROI_FADE:
                    prior_combined_mask = core.find_prior_combined_white_mask(prior_masks)
                    layer_index = get_numeric_part_func(filename)
                    rois = core.identify_rois(current_white_mask, config.roi_params.min_size)
                    classified_rois = tracker.update_and_classify(rois, layer_index, config)
                    receding_gradient = core.process_z_blending(
                        current_white_mask,
                        prior_combined_mask,
                        config,
                        classified_rois,
                    )
                else:
                    receding_gradient = _calculate_3d_gradient_field(
                        current_white_mask,
                        prior_masks,
                        config,
                    )

                output_image = core.merge_to_output(original_current_image, receding_gradient)
                A_out[z, :, :] = output_image

def create_output_array(array_path, height, width, num_layers):
    """Creates a 3D TileDB array to store the output image stack."""
    dom = tiledb.Domain(
        tiledb.Dim(name="Z", domain=(0, num_layers - 1), tile=1, dtype=np.uint32),
        tiledb.Dim(name="Y", domain=(0, height - 1), tile=height, dtype=np.uint32),
        tiledb.Dim(name="X", domain=(0, width - 1), tile=width, dtype=np.uint32),
    )
    schema = tiledb.ArraySchema(
        domain=dom,
        sparse=False,
        attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)],
    )
    tiledb.Array.create(array_path, schema)

def write_output_to_pngs(output_array_path, output_folder, image_filenames):
    """Reads the processed data from the output array and saves as PNG files."""
    with tiledb.open(output_array_path, 'r') as A:
        for i, filename in enumerate(image_filenames):
            img_data = A[i, :, :]["pixel_value"]
            output_filepath = os.path.join(output_folder, os.path.basename(filename))
            cv2.imwrite(output_filepath, img_data)


def run_tiledb_pipeline(config: Config, input_path: str, output_path: str, status_callback=None):
    """
    Orchestrates the TileDB processing pipeline.
    """
    if status_callback: status_callback("Starting TileDB pipeline...")

    input_folder = input_path
    output_folder = output_path

    if not input_folder or not os.path.isdir(input_folder):
        raise FileNotFoundError(f"Input folder not found or is not a directory: {input_folder}")

    with tempfile.TemporaryDirectory() as temp_dir:
        input_array_path = os.path.join(temp_dir, "input_array")
        output_array_path = os.path.join(temp_dir, "output_array")

        numeric_pattern = re.compile(r'(\d+)\.\w+$')
        def get_numeric_part(filename):
            match = numeric_pattern.search(filename)
            return int(match.group(1)) if match else float('inf')

        all_image_filenames = sorted(
            [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.bmp', '.tif', '.tiff'))],
            key=get_numeric_part
        )

        image_filenames_filtered = []
        for f in all_image_filenames:
            numeric_part = get_numeric_part(f)
            if config.start_index is not None and numeric_part < config.start_index:
                continue
            if config.stop_index is not None and numeric_part > config.stop_index:
                continue
            image_filenames_filtered.append(f)

        image_filepaths = [os.path.join(input_folder, f) for f in image_filenames_filtered]

        if not image_filepaths:
            if status_callback: status_callback("No PNG images found in the input folder.")
            return

        first_img = cv2.imread(image_filepaths[0], cv2.IMREAD_GRAYSCALE)
        height, width = first_img.shape
        num_layers = len(image_filepaths)

        if status_callback: status_callback("Creating TileDB array and ingesting images...")
        create_tiledb_array(input_array_path, height, width, num_layers)
        ingest_images_to_tiledb(input_array_path, image_filepaths, config.start_index, config.stop_index)

        if status_callback: status_callback("Processing Z-blending with TileDB...")
        process_and_write_results(input_array_path, output_array_path, config, image_filenames_filtered, get_numeric_part)

        if status_callback: status_callback("Writing output PNG files...")
        write_output_to_pngs(output_array_path, output_folder, image_filenames_filtered)

        if status_callback: status_callback("TileDB pipeline finished.")
