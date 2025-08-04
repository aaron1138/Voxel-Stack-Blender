Voxel Stack Blender / Euclidean Distance Blend Slices

An image processing tool designed for vertically (Z) blending large stacks of high-resolution black and white SLA/mSLA/resin printing slice images. It also features a slot based XY processing tab for application of grayscale LUTs critical for Z axis voxel accumulation during the resin printing process.

The first step is Z-axis blending built upon creating a blend gradient with the layer(s) below by way of a Euclidean distance map and masking operations.  Next is usually applying one of the non-linear grayscale LUTs to combat the logarithmic and strongly thresholded nature of voxel growth along the Z axis.  Then XY blending operations and additional LUT stacking are available for smoothing along the XY layer plane.  Resizing is also available for multi-sampling approaches, however the current version will not stack along the Z axis.  Prior efforts focused on Z-axis stacking for resolution enhancement and height blending. The Euclidean distance gradient has proven much faster, smoother, and better at retaining detail than direct layer stacking / sampling / blending. 

🛠️ Installation
Prerequisites: Python 3.8 or newer

Clone the repository to your local machine:

git clone https://github.com/aaron1138/Euclidean-Distance-Blend-Slices.git
cd Euclidean-Distance-Blend-Slices

Create a Python virtual environment and activate it:python -m venv venv

# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

Install the required dependencies:

pip install -r requirements.txt
Note: The requirements.txt file should contain: PySide6, numpy, opencv-python, and matplotlib.


🚀 Usage
The program is run from the main.py entry point.Open your terminal or command prompt in the project directory.Ensure your virtual environment is active.Run the application:python main.py

Using the Interface
Main Settings: Configure input/output folders and the core vertical blending parameters (N Layers, Fade Distance, Gamma, etc.).Processing Pipeline: This is where you build your custom workflow.Use the tabs at the bottom to create new LUT or XY Blend operations.Click the "Add to Pipeline" button to add a configured operation to the main list.Drag and drop items in the list to reorder them.Use the "Move Up/Down" and "Remove" buttons to manage the pipeline.Performance & Memory: Adjust multithreading and memory management settings, including the number of threads and the sliding window size.When you are ready to start, click the "Start Processing" button. The program will execute the steps in the order they appear in the pipeline list.

🤝 Contributing
Contributions are welcome! If you have suggestions for new features, bug fixes, or performance improvements, please open an issue or submit a pull request.

📄 License
Copyright 2025 Aaron Baca
GNU Affero General Public License
See license.txt for additional details.

































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