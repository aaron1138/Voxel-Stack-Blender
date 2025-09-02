import os
import abc
import re
import cv2
import tiledb
import numpy as np
from typing import List, Tuple, Dict, Any

class DataLoader(abc.ABC):
    """Abstract base class for data loading."""
    def __init__(self, input_path: str, image_filenames: List[str]):
        self.input_path = input_path
        self.image_filenames = image_filenames
        self.numeric_pattern = re.compile(r'(\d+)\.\w+$')

    def get_numeric_part(self, filename: str) -> int:
        match = self.numeric_pattern.search(filename)
        return int(match.group(1)) if match else float('inf')

    @abc.abstractmethod
    def setup(self, status_callback, progress_callback):
        pass

    @abc.abstractmethod
    def get_image_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, int]:
        pass

    @abc.abstractmethod
    def get_tiledb_uri(self) -> str:
        pass

    def __len__(self):
        return len(self.image_filenames)

class FileDataLoader(DataLoader):
    """Loads image data from individual files."""
    def setup(self, status_callback, progress_callback):
        # No setup needed for file-based loading
        pass

    def get_image_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, int]:
        filename = self.image_filenames[index]
        filepath = os.path.join(self.input_path, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None, None, filename, -1
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        layer_index = self.get_numeric_part(filename)
        return binary_img, img, filename, layer_index

    def get_tiledb_uri(self) -> str:
        return None

class TileDBDataLoader(DataLoader):
    """Loads image data from a TileDB array."""
    def __init__(self, input_path: str, image_filenames: List[str], temp_folder: str):
        super().__init__(input_path, image_filenames)
        self.temp_folder = temp_folder
        self.tiledb_uri = os.path.join(self.temp_folder, "tiledb_slice_store")
        self.array = None

    def setup(self, status_callback, progress_callback):
        status_callback("Setting up TileDB store...")
        if tiledb.array_exists(self.tiledb_uri):
            print(f"TileDB store already exists at {self.tiledb_uri}. Reusing.")
            self.array = tiledb.open(self.tiledb_uri, 'r')
            return

        first_image_path = os.path.join(self.input_path, self.image_filenames[0])
        img = cv2.imread(first_image_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape
        num_layers = len(self.image_filenames)

        dom = tiledb.Domain(
            tiledb.Dim(name="layer", domain=(0, num_layers - 1), tile=1, dtype=np.uint32),
            tiledb.Dim(name="height", domain=(0, height - 1), tile=height, dtype=np.uint32),
            tiledb.Dim(name="width", domain=(0, width - 1), tile=width, dtype=np.uint32),
        )
        schema = tiledb.ArraySchema(domain=dom, sparse=False, attrs=[tiledb.Attr(name="pixel_value", dtype=np.uint8)])
        tiledb.Array.create(self.tiledb_uri, schema)

        status_callback("Populating TileDB store...")
        with tiledb.open(self.tiledb_uri, 'w') as A:
            for i, filename in enumerate(self.image_filenames):
                filepath = os.path.join(self.input_path, filename)
                img_data = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                if img_data is not None:
                    A[i, :, :] = img_data
                if (i + 1) % 25 == 0 or (i + 1) == len(self.image_filenames):
                    progress = int(((i + 1) / len(self.image_filenames)) * 100)
                    status_callback(f"Populating TileDB: {i + 1}/{len(self.image_filenames)} slices stored.")
                    progress_callback(progress)

        self.array = tiledb.open(self.tiledb_uri, 'r')
        status_callback("TileDB setup complete.")

    def get_image_data(self, index: int) -> Tuple[np.ndarray, np.ndarray, str, int]:
        filename = self.image_filenames[index]
        original_image = self.array[index]["pixel_value"]
        _, binary_image = cv2.threshold(original_image, 127, 255, cv2.THRESH_BINARY)
        # In the TileDB case, the layer_index is simply the array index.
        return binary_image, original_image, filename, index

    def get_tiledb_uri(self) -> str:
        return self.tiledb_uri

    def __del__(self):
        if self.array is not None:
            self.array.close()
