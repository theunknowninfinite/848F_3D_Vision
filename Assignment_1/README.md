# CMSC848F - Assignment 1: Rendering Basics with PyTorch3D


Goals: In this assignment, you will learn the basics of rendering with PyTorch3D,
explore 3D representations, and practice constructing simple geometry.

A report has been done, `report/starter.md.html`.

## Folder Struture
+ data
+ output_images
+ report
+ starter

## 0. Setup

You will need to install Pytorch3d. See the directions for your platform
[here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md).
You will also need to install Pytorch. If you do not have a GPU, you can directly pip
install it (`pip install torch`). Otherwise, follow the installation directions
[here](https://pytorch.org/get-started/locally/).

Other miscellaneous packages that you will need can be installed using the
`requirements.txt` file (`pip install -r requirements.txt`).

### Points to Note
+ The outputs are stored under `output_images`
+ All programs can be run from directly within VSCode or using a terminal.
+ Ensure you are in the root dictionary when running the programs.

## Rendering first mesh

Here, the renderer is abstracted using the `get_mesh_renderer` wrapper function in `utils.py`.
The code to render the mesh can be run by:
```bash
python render_mesh.py
```

## 1. Practicing with Cameras

### 1.1. 360-degree Renders

The code to render the mesh can be run by:
```bash
python render_360.py
```
### 1.2 Re-creating the Dolly Zoom

The code to render the mesh can be run by:
```bash
python dolly_zoom.py
```

## 2. Practicing with Meshes

The code to render the mesh can be run by:
```bash
python color_change.py
```
### 2.1 Constructing a Tetrahedron

The code to render the mesh can be run by:
```bash
python render_tetra.py
```
### 2.2 Constructing a Cube

The code to render the mesh can be run by:
```bash
python render_cube.py
```

## 3. Re-texturing a mesh

The code to render the mesh can be run by:
```bash
python color_change.py
```
## 4. Camera Transformations
The code to render the mesh can be run by:
```bash
python camera_transforms.py
```
## 5. Rendering Generic 3D Representations

### 5.1 Rendering Point Clouds from RGB-D Images

The code to render the point cloud 1 can be run by:
```bash
python -m render_generic --render rgbd1
```
For rendering point cloud 2 and point cloud union,
change the flag `-- render` to `rgbd2` or `rgbdunion`.

### 5.2 Parametric Functions

The code to render the point cloud can be run by:
```bash
python -m render_generic --render toruspts
```

### 5.3 Implicit Surfaces
The code to render the mesh can be run by:
```bash
python -m render_generic --render torus
```
## Additional Arguments

+ The input for the cow mesh can be changed by passing the flag `--cow_path` with path of the file.Default location is under `\data`

+ The image size can be changed by passing the flag `--image_size` with apt size.

+ The output path to save the gifs for `render_generic.py` can be changed by passing the flag `--ouput_path` with the file location.

## Credits
[848F-Assignment 1](https://github.com/848f-3DVision/assignment1) for parts readme file and starter code.
Pytorch 3D documentation
[Torus Equations](https://mathworld.wolfram.com/Torus.html)