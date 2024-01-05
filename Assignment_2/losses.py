import torch
from pytorch3d.ops import *
from pytorch3d.loss import mesh_laplacian_smoothing
import pytorch3d.loss
import pytorch3d
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	loss_fn = torch.nn.BCELoss()
	loss=loss_fn(torch.sigmoid(voxel_src),voxel_tgt)
	# implement some loss for binary voxel grids
	return loss

def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3
	# loss_chamfer =
	# implement chamfer loss from scratch
	x_dist=knn_points(point_cloud_src,point_cloud_tgt)
	x_knn_dist= x_dist.dists[...,0]
	y_dist=knn_points(point_cloud_tgt,point_cloud_src)
	y_knn_dist= y_dist.dists[...,0]
	x_mean= torch.mean(x_knn_dist)
	y_mean=torch.mean(y_knn_dist)
	loss_chamfer= x_mean+y_mean
	# loss_test = pytorch3d.loss.chamfer_distance(point_cloud_src,point_cloud_tgt)
	# print("CHAMFER LOSS",loss_test)
	return loss_chamfer

def smoothness_loss(mesh_src):
	loss_laplacian = mesh_laplacian_smoothing(mesh_src)
	# implement laplacian smoothening loss
	return loss_laplacian
