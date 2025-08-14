import os
import re
import subprocess
from typing import List

def extract_layers(uvtools_path: str, input_file: str, temp_folder: str) -> str:
    """
    Executes UVToolsCmd.exe to extract layers into a timestamped temp folder.
    Returns the path to the folder containing the extracted images.
    """
    input_folder = os.path.join(temp_folder, "Input")
    os.makedirs(input_folder, exist_ok=True)

    command = [uvtools_path, "extract", input_file, input_folder, "--content", "Layers"]

    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)
        if process.returncode not in [0, 1]:
            raise RuntimeError(f"UVTools exited with an error (code {process.returncode}):\n\n{process.stderr}")
        return input_folder
    except Exception as e:
        raise RuntimeError(f"UVTools extraction failed: {e}")

def generate_uvtop_file(processed_images_folder: str, temp_folder: str, run_timestamp: str) -> str:
    """Generates the .uvtop XML file for repacking."""
    numeric_pattern = re.compile(r'(\d+)\.\w+$')
    def get_numeric_part(filename):
        match = numeric_pattern.search(filename)
        return int(match.group(1)) if match else float('inf')

    processed_files = sorted(
        [os.path.join(processed_images_folder, f) for f in os.listdir(processed_images_folder) if f.lower().endswith('.png')],
        key=get_numeric_part
    )

    if not processed_files:
        raise RuntimeError("No processed image files found to generate .uvtop file.")

    xml_content = '<?xml version="1.0" encoding="utf-8" standalone="no"?>\n'
    xml_content += '<OperationLayerImport xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
    xml_content += '  <LayerRangeSelection>None</LayerRangeSelection>\n'
    xml_content += '  <ImportType>Replace</ImportType>\n'
    xml_content += '  <Files>\n'
    for f_path in processed_files:
        xml_content += '    <GenericFileRepresentation>\n'
        xml_content += f'      <FilePath>{f_path}</FilePath>\n'
        xml_content += '    </GenericFileRepresentation>\n'
    xml_content += '  </Files>\n'
    xml_content += '</OperationLayerImport>\n'

    uvtop_filename = f"repack_operations_{run_timestamp}.uvtop"
    uvtop_filepath = os.path.join(temp_folder, uvtop_filename)

    with open(uvtop_filepath, 'w', encoding='utf-8') as f:
        f.write(xml_content)

    return uvtop_filepath

def repack_layers(uvtools_path: str, input_file: str, uvtop_filepath: str, output_location: str, temp_folder: str, output_prefix: str, run_timestamp: str) -> str:
    """Executes UVToolsCmd.exe to repack the processed layers."""
    original_filename = os.path.basename(input_file)
    output_filename = f"{output_prefix}{run_timestamp}_{original_filename}"

    output_directory = ""
    if output_location == "input_folder":
        output_directory = os.path.dirname(input_file)
    else: # Default to working_folder
        output_directory = temp_folder

    final_output_path = os.path.join(output_directory, output_filename)

    command = [uvtools_path, "run", input_file, uvtop_filepath, "--output", final_output_path]

    try:
        creation_flags = subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
        process = subprocess.run(command, capture_output=True, text=True, check=True, creationflags=creation_flags)
        if process.returncode not in [0, 1]:
            raise RuntimeError(f"UVTools exited with an error (code {process.returncode}):\n\n{process.stderr}")
        return final_output_path
    except Exception as e:
        raise RuntimeError(f"UVTools repacking failed: {e}")
