import os
import tiledb
import numpy as np
import cv2
from config import Config
import processing_core as core
import re

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

def process_and_write_results(input_array_path, output_array_path, config: Config):
    """
    Processes Z-blending using data from the TileDB array and writes to an output array.
    """
    with tiledb.open(input_array_path, 'r') as A_in:
        num_layers, height, width = A_in.schema.domain.shape

        if tiledb.array_exists(output_array_path):
            tiledb.remove(output_array_path)

        create_output_array(output_array_path, height, width, num_layers)

        with tiledb.open(output_array_path, 'w') as A_out:
            for z in range(num_layers):
                if z < config.receding_layers:
                    # Not enough prior layers, just write the original image
                    A_out[z, :, :] = A_in[z, :, :]["original"]
                    continue

                current_white_mask = A_in[z, :, :]["binary"]
                original_current_image = A_in[z, :, :]["original"]

                start_z = max(0, z - config.receding_layers)
                prior_masks_data = A_in[start_z:z, :, :]["binary"]

                prior_masks = [prior_masks_data[i, :, :] for i in range(prior_masks_data.shape[0])]

                receding_gradient = core.process_z_blending(
                    current_white_mask,
                    prior_masks,
                    config,
                    classified_rois=[], # ROI mode not supported in this example
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


def run_tiledb_pipeline(config: Config, status_callback=None):
    """
    Orchestrates the TileDB processing pipeline.
    """
    if status_callback: status_callback("Starting TileDB pipeline...")

    input_folder = config.input_folder
    output_folder = config.output_folder

    # Create a temporary directory for the TileDB arrays
    temp_dir = os.path.join(output_folder, "tiledb_temp")
    os.makedirs(temp_dir, exist_ok=True)
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
    process_and_write_results(input_array_path, output_array_path, config)

    if status_callback: status_callback("Writing output PNG files...")
    write_output_to_pngs(output_array_path, output_folder, image_filenames_filtered)

    # Clean up temporary TileDB arrays
    if tiledb.array_exists(input_array_path):
        tiledb.remove(input_array_path)
    if tiledb.array_exists(output_array_path):
        tiledb.remove(output_array_path)
    if os.path.exists(temp_dir):
        os.rmdir(temp_dir)

    if status_callback: status_callback("TileDB pipeline finished.")
