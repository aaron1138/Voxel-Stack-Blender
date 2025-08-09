## Voxel Stack Blender / Euclidean Distance Slice Blender

A tool designed for improved Z-axis blending[^1] and smoothing of mSLA / SLA resin printing slice files. This also features an expanded toolset of XY blending and smoothing post processors as well as gray scale remapping functions to match voxel growth response to anisotropic resin printing dimensions and conditions. 

We start with Z-axis blending built upon generating a grayscale gradient of the current working layer with the layer(s) below using a Euclidean distance map and masking operations.  Next is usually applying one of the non-linear grayscale LUTs to combat the logarithmic and strongly thresholded nature of voxel growth along the Z axis.  Then XY blending operations and additional LUT operation stacking are available for smoothing along the XY layer plane.  Resizing is also available for multi-sampling approaches, however the current version will not merge layers along the Z axis.  Prior efforts focused on Z-axis stack merging and sampling for resolution enhancement and height blending. The Euclidean distance gradient has proven much faster, smoother output, and better at retaining detail than direct layer stacking / sampling / blending. 

The Python source here was primarily composed working with LLMs / Generative AI based on the algorithms and general math concepts of interest to the "author".  Yeah, this is vibe code, but I knew the math of what I wanted it to do, so that counts, right?  I'm not a programmer and it's been a while since my "Introduction to C Programming" (not even C++) in college.

Performance is not too bad in Python thanks to Numpy being C under the covers, but if someone would like to port the functionality to native UVTools C# scripting, I would be happy to help.

### Recent Results (Aug 5th Update)

![Heads Up Comparison CB Blur 6 vs VBlend Preset - Gale - HSBase - LD4-40px-upShiftedExp(LUT)-Bilateral-PiecewiseLUT](https://github.com/user-attachments/assets/7b0585ce-82d5-4984-865a-40432950c26c)

 - Excellent results as seen above with just a couple tweaks to the processing parameters.  Takes around 3-7 minutes to process on a higher end PC depending on how many cores you toss at it including UVTools extraction and repackaging of slice file. 
 - Input for Voxel-Stack-Blender from Chitubox **not anti-aliased** as per script input requirements.
 - Preset configuration for those results added, `presets/Preset-Double-LUT.json`.  
 - Preset requires user to go to the first LUT and reset the mapping to `saved_luts/EXP(LUT)-upShifted.json`.
 - More details of the preset below.




### 🛠️ Installation
Prerequisites: Python 3.8 or newer

Clone the repository to your local machine:<br/>
`git clone https://github.com/aaron1138/Voxel-Stack-Blender.git`<br/>
`cd Voxel-Stack-Blender`<br/>

Create a Python virtual environment and activate it:<br/>
`python -m venv venv`<br/>

On Windows:<br/>
`.\venv\Scripts\activate`<br/>
On macOS/Linux:<br/>
`source venv/bin/activate`<br/>

Install the required dependencies:<br/>
`pip install -r requirements.txt`<br/>



### 🚀 Usage
With you Python virutal environment active run  `python main.py`.

**2025-08-04** 
 - Added UVTools direct integration for extraction and repacking of slice files via `uvtoolscmd.exe`.  I'll rewrite this readme later for more exact process details, but steps 1, 2, and 6 are effectively not required and the alternate UVT steps should be intuitive.  
 - I would recommend still using the "folder" process a few times and manually scrubbing through the layers output inside UVTools so you can see the results and better understand the grayscale blending being applied.
 - Some files and model shapes produce odd results.  Anything with a flat horizontal surface which then has multiple vertical protrusions gets heavy fillets around the protrusions.  Odd looking in the slice files for rafts*, but probably makes they stronger.  Outright weird results on flat exposure and resin feature tests.
   - *For issues with rafts having too many gray pixels, you could swap back in early layers from the original slice file with UVTools.  I have not observed this being an issue though.  My brushes with the lasagna bug have been a couple dozen millimeters up in the print.  It seems the gray cones around supports isn't a problematic amount of entropy at least for the S4U. 
 - Added Input & Output Min/Max adjustments for the Generate LUTs.
   - Allows creation of piecewise LUTs.
   - Input acts a filter where gray values outside the range will not be touched (needed to pass through 0 black frequently).
   - Output acts as a compression range allowing curve shifting. 
   - Should add a button to save a set of LUTs in a row as a single file LUT. No optimization to concatenate multiple consequetive LUTs. 

1. Use UVTools or similar to extract PNGs of your slices numbered to a folder from your slice file.  `File -> Extract file contents` or `<Ctrl>+<Shift>+E`
   - Recommended: slice files with NO anti-aliasing.  
     - The first stage Euclidean distance blending only looks at black and white pixels.  
     - Baked in gray pixels will result in odd gray halos.
   - Use padded numbering (e.g. 0001.png) and remove any extraneous files such as print previews (3d.png, preview.png, etc.) and print parameters (usually ini/json/txt) is also recommended. 
     - It should usually recognize and handle prefixes, padding, and unpadded naturally numbered files (e.g. 1.png, 2.png). 
     - NanoDLP and other unusually formatted PNG files may need an additional step due to their odd use of 3-channel grayscale images at 1/3rd resolution. 
2. Configure input/output folders. Creation of a separate folder for output is usually recommended.
3. Recommended: Set the number of threads you want to use.  This controls both speed and memory utilization.  
   - A single 12k slice file needs 50MiB of RAM once uncompressed before we even touch actual processing, floating point upconversions, and mask. As we`re working with several slices per worker thread along with equally dimensioned maskes and float arrays in Python. This blows up quick and this code is not "production ready" optimized.  
   - Processing 12 threads of 12k images with a look down of 4 will vary between 4-8GiB of RAM utilization as threads enter and exit along with the sliding window slice handler.
4. Configure the Z-axis Blend Parameters:
   - "Look Down N Layers" - 2-4 is usually good.  Each layer will "look down" at this many preceding layers to see if it is receding along any edge from those N layers below.
   - Recommended: Enable "Use Fixed Fade Distance" with a number of pixels which will control the fade gradient.  Tests worked well with 15-40.
5. Configure the XY Blend Pipeline.  These steps are executed for each slice after the gradient is applied and the 8-bit grayscale result of the Euclidean Distance blending is applied.
   - Recommend starting with a Z-axis growth compensating Apply LUT operation.  The included `EXP(LUT)-upShifted.json` has been put together based on the the high threshold of 40-60% gray necessary for any Z layer voxel growth to begin as well as the natural log / exponential curve of SLA resin voxel growth.
   - Next a blending operation, usually a Gaussian Blur is good now to add some interlayer smoothing.  Kernel / matrix sizes are odd numbered rather than a "radius" setting.  Between the kernel size settings and separate X & Y sigma values, you can compensate for anisotropy of voxel XY dimensions (a bit overkill for most).
   - **Why Z-LUT then blur XY?** -- XY grayscale can grow laterally in the XY plane when touching adjacent lit pixels and with much higher sub-voxel resolution than Z growth of lone voxels.  This allows additional contour matching to the horizontal features of the layer.   
   - Additional blending and LUT options may be stacked in the slots for further effect.
   - Resize operations are also available for those rendering slices at higher (or lower I suppose) resolutions than their printer accepts.
6. Using UVTools or similar, repack your slices in the orignal slice file (or a copy) using the `Actions > Import Layers` choosing `Import type: Replace...`.  Save your file and send it to your printer as normal.
   - Note: Prints with as thick a grayscale exterior as this process may produce will be softer than normal prior to post curing.  They are more easily damaged during the wash process.  This is visible on some example prints in the `images/` folder as pitting from ultrasonic cavitation and minor gouging down towards the 100% white layers. 


### More Details about this first *works nice enought to share* Preset:

 - Details of the preset / config (order matters for the XY Ops):
   - Currently using 40um layers with Anycubic Texture Gray Resin.
   - Look Down 4 layers / Fixed Distance Fade 40 px 
   - Apply LUT -> File -> EXP(LUT)-upShifted.json
   - Bilateral Filter, diameter 7, Sigma Color 60, Sigma Space 60
   - Gaussian Blur, kernel 3x3, Sigma (X, Y) 0.8, 0.8
   - Apply LUT -> Generated -> Input Min / Max 200-255, Output Min/Max 200-255, Param 2.00. *The sauce here it is brings back down bright whites diffused out by the middle 2 filters.  If you look at the slice files the bottom of the curve (200) looks like a lighter halo, but that is just the optical illusion from a few pixel band of gray the same color instead of darkening.*
   - Shows the utility of the segmented / clipped generation setup for LUTs, pre-baked LUTs like EXP(LUT)-upShifted can be pretty closely copied in 1 or 2 piecewise LUTs.
   

### ⚠️ Warnings and Advisories
 - This software comes with no warranties.  Print / slice file corruption and printer defects in handling gray pixels may cause **physical and mechanical damage to your printer**. 
 - This software has no respect for your RAM or other resources.  Use care with selecting Thread count and other options.
 - This program can produce a very high number of gray pixels which is a known bug for most consumer mSLA resin printers using Chitu mainboards. This can result in "lasagna bug" corruption, missing layers, missing gray pixels on random layers[^2], or if you are lucky, just very slow image loading before exposure (supposedly older Mars/Saturns).  
 - Some layers will look a bit *odd*. This especially happens when you have a flat layer in the XY plane which abruptly has protrusions (like rafts).  Nothing to worry about for rafts.  For flat exposure / RP tests, it builds some odd fillets around objects. 
 - Used aggressively, Gaussian Blur and similar in the XY blend pipline can weaken supports, especially contact points.  This is because (a) we are not able to identify supports from the model in the sliced file and (b) unlike slicer blur operations, we are doing a symmetric operation with Gaussian blur without an outward dilation of the edge. 
 - Like any other grayscale smoothing this reduces some detail.  I have tried to give as much control via all the nerd knobs as possible without building a slicer from the ground up (don`t currently have those skills - if you do and want to consult, let me know)
 - Increased Rest After Retract / Wait time before cure of 2s for standard layers and a clean has been successful[^3] to help image loading with sparsely filled build plates (i.e. 6-10 minis) on my Saturn 4 Ultra.  I am curious to hear others` findings. 
  
 [^1]: Ummm, legally distinct from vertical and/or 3D anti-aliasing or something like that...
 [^2]: Observed on Saturn 4 Ultra 12k with Dec 2024 or Mar 2025 firmware.  I suppose that is better than lasagna. 
 [^3]: The above missing gray pixels every 4th layer or so lead to increasing rest / wait time mentioned above to mitigate




### 🤝 Contributing
 - Contributions are welcome! If you have suggestions for new features, bug fixes, or performance improvements, please open an issue or submit a pull request.
 - It would be great to port this to UVTools C# or similar higher performance scripting.
 



### 👉 Useful links to additional information:
- UVTools: https://github.com/sn4k3/UVtools
- Richard Greene / Autodesk Ember team research on High Fidelity / Sub-voxel SLA resin printing (slicers are very behind). 
    - https://www.youtube.com/watch?v=PsK7An7ymYk
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
