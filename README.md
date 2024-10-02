# Grasp Quality Dataset Generator for Bin Picking

## Overview

This project is a Python-based tool integrated with Blender that generates synthetic datasets for bin-picking tasks. It uses Blender's physics engine and various simulation settings to create scenes where objects are placed in bins, and a grasp quality (GQ) map is generated for each scene.

The dataset includes RGB images, depth maps, normal vectors, and segmentation masks, with an additional Grasp Quality map that measures the suitability of each pixel for grasping.

This tool is primarily aimed at researchers or engineers working on robotic grasping tasks and machine learning models for bin-picking.

## Install Instructions

Install Required Libraries:

Before running the script, ensure the following libraries are installed in Blender’s Python environment. Blender uses its own Python, so you’ll need to install these packages through Blender’s Python.

First, locate Blender’s Python executable, typically found in:

- **Windows**: C:\Program Files\Blender Foundation\Blender <version>\<version>\python\bin\python.exe
- **Mac**: /Applications/Blender.app/Contents/Resources/<version>/python/bin/python3.7m
- **Linux**: /path_to_blender/<version>/python/bin/python3.x

Once you have the path to Blender's Python, open a terminal (or command prompt) and use pip to install the required libraries:

```bash
<path_to_blender_python> -m ensurepip
<path_to_blender_python> -m pip install pandas numpy matplotlib opencv-python OpenEXR Imath Pillow scikit-image scipy scikit-learn imageio
```

Download the Script:

Download the `main.py` file from this repository.

Install as a Blender Add-on:

1. Open Blender.
2. Go to **Edit > Preferences > Add-ons**.
3. Click **Install...** and select the downloaded `main.py` file.
4. Enable the add-on by checking the box next to its name.


### Usage Instructions:

Once you have installed and enabled the add-on, follow these steps to generate the Grasp Quality (GQ) dataset:

1. **Open the Tool Panel**:
   - In the 3D Viewport, press `N` to open the side panel.
   - You should see a new tab named "GQ Dataset Generator."

2. **Set Paths for Object and Bin Directories**:
   - In the "Model selection" section:
     - Set the **Objects Directory**: This should be the directory containing your 3D objects (.obj, .stl, .ply files).
     - Set the **Bin Directory**: This should contain the bin models where objects will be placed.

3. **Material JSON File**:
   - In the **Material JSON** section, select the path to the JSON file that contains material properties for rendering.

4. **Configure Scene Parameters**:
   - In the "Scene Parameters" panel, you can configure the following:
     - **Bin Scaling**: Adjust the scale for the bin models.
     - **Object Quantity**: Set the minimum and maximum number of objects per scene.
     - **Object Scaling**: Enable scaling and set the range for object sizes.
     - **Lighting**: Set the number of lights, light strength, and whether to use random-colored lights.
     - **Camera Settings**: Adjust the camera’s maximum angle, height, resolution, and focal length. You can also enable stereo images.

5. **Output Settings**:
   - In the "Output" panel, choose which data to generate for each scene:
     - **RGB Images**, **Depth Maps**, **Normal Vectors**, **Segmentation Masks**, and **Grasp Quality Maps**.
   - Specify the output directory for the dataset and the size of the dataset (number of scenes).
   - You can also enable the generation of **GQ Elements** and **Position Data** for the generated scenes.

6. **Start the Dataset Generation**:
   - Once all parameters are set, click the **Generate Dataset** button at the top of the panel.
   - The script will now generate scenes based on the given settings. Each scene will have the selected output (RGB, Depth, etc.) saved in the specified output directory.

7. **Access Output Files**:
   - After the dataset generation is complete, the output files (images, maps, and CSVs) will be stored in the chosen output directory.
   - These files will be organized into subfolders based on the type of data (e.g., `rgb`, `depth`, `normals`, `segmentation`, `gq`, etc.).

8. **Optional Features**:
   - **Stereo Images**: If enabled, stereo pairs (left and right images) will be generated.
   - **Bounding Boxes**: If segmentation is enabled with instance mode, you can generate bounding boxes for objects in the scenes.
