## Voxel Stack Blender / Euclidean Distance Slice Blender

An image processing tool designed for improved Z-axis blending and smoothing of mSLA / SLA resin printing slice files. This also features an expanded toolset of XY blending and smoothing post processors as well as gray scale remapping functions to match voxel growth response to anisotropic resin printing dimensions and conditions. 

We start with Z-axis blending built upon generating a grayscale gradient of the current working layer with the layer(s) below using a Euclidean distance map and masking operations.  Next is usually applying one of the non-linear grayscale LUTs to combat the logarithmic and strongly thresholded nature of voxel growth along the Z axis.  Then XY blending operations and additional LUT operation stacking are available for smoothing along the XY layer plane.  Resizing is also available for multi-sampling approaches, however the current version will not merge layers along the Z axis.  Prior efforts focused on Z-axis stack merging and sampling for resolution enhancement and height blending. The Euclidean distance gradient has proven much faster, smoother output, and better at retaining detail than direct layer stacking / sampling / blending. 

The Python source here was primarily composed by LLMs / Generative AI based on the algorithm and general math concepts described by the "author".  Yeah, this is all vibe coded, but I knew the math of what I wanted it to do, so that counts, right?

### 🛠️ Installation
Prerequisites: Python 3.8 or newer

Clone the repository to your local machine:

`git clone https://github.com/aaron1138/Voxel-Stack-Blender.git`
`cd Voxel-Stack-Blender`

Create a Python virtual environment and activate it:

`python -m venv venv`

On Windows:

`.\venv\Scripts\activate`

On macOS/Linux:

`source venv/bin/activate`

Install the required dependencies:

`pip install -r requirements.txt`


### 🚀 Usage
The program is run from the main.py. Open your terminal or command prompt in the project directory. Ensure your virtual environment is active if in use. Run the application: 'python main.py'.

1) Use UVTools or similar to extract PNGs of your slices numbered to a folder from your slice file.  'File -> Extract file contents' or '<Ctrl>+<Shift>+E'
 - Recommended: slice files with NO anti-aliasing.  The first stage Euclidean distance blending only looks at black and white pixels.  Baked in gray pixels will result in odd gray halos.
 - Using padded numbering and removing any extraneous files such as print previews (3d.png, preview.png, etc.) and print parameters (usually ini/json/txt) is recommended, but it should usually recognize and handled prefixes, padding, and unpadded naturally numbered files. NanoDLP files may need an additional step due to their odd use of 3-channel grayscale images at 1/3rd resolution. 
2) Configure input/output folders. Creation of separate folders for output is usually recommended.
3) Recommended: Set the number of threads you want to use.  This controls both speed and memory utilization.  A single 12k slice file needs 50MiB of RAM once uncompressed before we even touch actual processing, floating point upconversions, and mask. As we're working with several slices per worker along with equally dimensioned maskes and float arrays in Python, this blows up quick.  Processing 12 threads of 12k images with a look down of 4 will vary between 4-8GiB of RAM utilization as threads enter and exit along with the sliding window slice handler.
4) Configure the Z-axis Blend Parameters:
 - "Look Down N Layers" - 2-4 is usually good.  Each layer will "look down" at this many preceding layers to see if it is receding along any edge from those N layers below.
 - Recommended: Enable "Use Fixed Fade Distance" with a number of pixels which will control the fade gradient.  Tests worked well with 15-40.
5) Configure the XY Blend Pipeline.  These steps are executed for each slice after the gradient is applied and the 8-bit grayscale result of the Euclidean Distance blending is applied.
 - Usually recommend starting with a Z-axis growth compensating Apply LUT operation.  The included 'EXP(LUT).json has been put together based on the the high threshold of 40-60% gray necessary for any layer growth to start as well as the natural log / exponential curve of SLA resin voxel growth.
 - Next a blending operation, usually a Gaussian Blur is good to now add some interlayer smoothing.  Kernel / Matrix sizes are odd numbered rather than a "radius" setting.  Between the kernel size settings and separate X & Y sigma values, you may compensate for anisotropy of voxel XY dimensions (a bit overkill for most).  
 - Additional blending and LUT options may be stacked in the slots for further effects.
 - Resize is also available for those rendering slices at higher (or lower I suppose) resolutions than their printer accepts.
 6) Using UVTools or similar, repack your slices in the orignal slice file (or a copy) using the 'Actions > Import Layers' choosing 'Import type: Replace...'.  Save your file and send it to your printer as normal.


### ⚠️ Warnings and Advisories
 - This software comes with no warranties.  Print / slice file corruption and printer defects in handling gray pixels may cause **physical and mechanical damage to your printer**. 
 - This program can produce a very high number of gray pixels which is a known bug for most consumer mSLA resin printers using Chitu mainboards. This can result in "lasagna bug" corruption, missing layers, missing gray pixels on random layers*, or if you are lucky, just very slow image loading before exposure (supposedly older Mars/Saturns).  
 - Some layers will look a bit *odd*. This especially happens when you have a flat layer in the XY plane which abruptly has protrusions (like rafts).  Nothing to worry about for rafts.  For flat exposure / RP tests, it builds some odd fillets around objects. 
 - Like any other grayscale smoothing this reduces some detail.  I have tried to give as much control via all the nerd knobs as possible without building a slicer from the ground up (don't currently have those skills - if you do and want to consult, let me know)
 - Increased Rest After Retract / Wait time before cure of 2s for standard layers and a clean has been successful to help image loading with sparsely filled build plates (i.e. 6-10 minis) on my Saturn 4 Ultra.  I am curious to hear others' findings. 
  
 *(Observed on Saturn 4 Ultra 12k with Dec 2024 or Mar 2025 firmware.  I suppose that is better than lasagna. This lead to increasing rest / wait time mentioned above to mitigate)

### 🤝 Contributing
Contributions are welcome! If you have suggestions for new features, bug fixes, or performance improvements, please open an issue or submit a pull request.

### 👉 Useful links to additional information:
- Richard Greene / Autodesk Ember team research on High Fidelity / Sub-voxel SLA resin printing (slicers are very behind). 
    - https://www.youtube.com/watch?v=PsK7An7ymYk
- Resin printing VOCs are vastly less dangerous than portrayed on Reddit and Discord by those gatekeeping via safety and fear.  This study focuses specifically on home and small office (e.g. dental) printing scenarios. 
    - https://www.nature.com/articles/s41370-025-00778-y
    - For those unfamiliar with reading academic studies, that whole conclusion block is a politely worded condemnation of safety fear mongering. 
- "Lasagna bug" demonstrated with Saturn 4 Ultra (older firmware??) 
    - https://www.youtube.com/watch?v=E5PAmhOnDps


### 📄 License
Copyright 2025 Aaron Baca

GNU Affero General Public License

https://www.gnu.org/licenses/agpl-3.0-standalone.html

See license.txt for a copy of the above license text and details.

```
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
```