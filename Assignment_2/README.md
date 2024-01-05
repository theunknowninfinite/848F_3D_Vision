# CMSC848F Assignment 2: Single View to 3D

Goals: In this assignment, you will explore the types of loss and decoder functions for regressing to voxels, point clouds, and mesh representation from single view RGB input.

## Contents of Zips
1. Code
2. Report

## 0. Setup

Please download and extract the dataset from [here](https://drive.google.com/file/d/1VoSmRA9KIwaH56iluUuBEBwCbbq3x7Xt/view?usp=sharing).
After unzipping, set the appropiate path references in `dataset_location.py` file

Make sure you have installed the packages mentioned in `requirements.txt`.
This  will need the GPU version of pytorch.

## 1. Exploring loss functions (15 points)
This section will involve defining a loss function, for fitting voxels, point clouds and meshes.

### 1.1. Fitting a voxel grid (5 points)

Run the file `python fit_data.py --type 'vox'`, to fit the source voxel grid to the target voxel grid.The render will automatically be done in a directory under output images with names `voxel_*.gif*`.


### 1.2. Fitting a point cloud (5 points)
Run the file `python fit_data.py --type 'point'`, to fit the source point cloud to the target point cloud.The render will automatically be done in a directory under output images with names `pc*.gif`.



### 1.3. Fitting a mesh (5 points)

Run the file `python fit_data.py --type 'mesh'`, to fit the source mesh to the target mesh.The render will automatically be done in a directory under output images with names `*_mesh.gif`.


## 2. Reconstructing 3D from single view (85 points)
This section will involve training a single view to 3D pipeline for voxels, point clouds and meshes.

We also provide pretrained ResNet18 features of images to save computation and GPU resources required. Use `--load_feat` argument to use these features during training and evaluation. This should be False by default, and only use this if you are facing issues in getting GPU resources. You can also enable training on a CPU by the `device` argument. Also indiciate in your submission if you had to use this argument.

### 2.1. Image to voxel grid (20 points)


Run the file `python train_model.py --type 'vox'`, to train single view to voxel grid pipeline.
The tuned hypoerparameters are  **iterations = 10000 batch size=1 Num_workers=4 .**
After trained, visualize the input RGB, ground truth voxel grid and predicted voxel in `eval_model.py` file using:
`python eval_model.py --type 'vox' --load_checkpoint` with batch size as 1.
Outputs are saved under `/output_images/vox`
The gifs are automatically made and seved for every 50 iterations of program being run.

### 2.2. Image to point cloud (20 points)
Run the file `python train_model.py --type 'point'`, to train single view to pointcloud pipeline.
The tuned hyper parameters are **N_points = 10000 iterations = 5000 batch size=1 Num_workers=4.**

After trained, visualize the input RGB, ground truth point cloud and predicted  point cloud in `eval_model.py` file using:
`python eval_model.py --type 'point' --load_checkpoint` with batch size as 1 and  N_points as 10000.
Outputs are saved under `/output_images/point`
The gifs are automatically made and seved for every 50 iterations of program being run.

### 2.3. Image to mesh (20 points)

<!-- In this subsection, we will define a neural network to decode mesh.

Similar as above, define the decoder network [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L177) in `model.py` file, then reference your decoder [here](https://github.com/848f-3DVision/assignment2/blob/main/model.py#L220) in `model.py` file -->

Run the file `python train_model.py --type 'mesh'`, to train single view to mesh pipeline.
The tuned hyperparameters are **N_points = 5000 iterations = 10000 batch size=1 Num_workers=4 w_smooth=1 w_chamfer=1**.

After trained, visualize the input RGB, ground truth mesh and predicted mesh in `eval_model.py` file using:
`python eval_model.py -h-type 'mesh' --load_checkpoint` with batch size as 1 and n_points as 5000.

Outputs are saved under `/output_images/mesh`
The gifs are automatically made and seved for every 50 iterations of program being run.

### 2.4. Quantitative comparisions(10 points)
The F1 scores have been compared and have been explained in the webpage attached.

### 2.5. Analyse effects of hyperparms variations (5 points)
The results and analysis of results has been done in the webpage attached.

### 2.6. Interpret your model (10 points)
The interpretation has been explained in the webpage attached.

## NOTE

1. Only the outputs used in the webpage are attached to the code to reduce size of package.
2. Ignore warnings about depricated functions.
3. The input image, ground truth and predicted values are automatically saved.
4. Define path in  `dataset_location.py` according to windows or linux host being hosted as the file path automatically changes depends on that.
5. The webpage that is the report can be found under the reports folder.