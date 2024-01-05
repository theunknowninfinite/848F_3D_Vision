"""
Sample code to render various representations.

Usage:
    python -m starter.render_generic --render point_cloud  # 5.1
    python -m starter.render_generic --render parametric  --num_samples 100  # 5.2
    python -m starter.render_generic --render implicit  # 5.3
"""
import argparse
import pickle

import matplotlib.pyplot as plt
import mcubes
import numpy as np
import pytorch3d
import torch
import imageio
from starter.utils import get_device, get_mesh_renderer, get_points_renderer, unproject_depth_image

def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_plant(verts,rgb,image_size=512,
    background_color=(1, 1, 1),
    device=None,):
    if device is None:
            device = get_device()
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color
    )
    verts = verts.to(device).unsqueeze(0)
    rgb = rgb.to(device).unsqueeze(0)
    point_cloud = pytorch3d.structures.Pointclouds(points=verts, features=rgb)
    list_of_images=[]
    azimuth=np.linspace(0,360,100)
    for i in azimuth:
        R, T = pytorch3d.renderer.look_at_view_transform(6, 10, i, up=((0, -1, 0),))
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud, cameras=cameras)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        
        from PIL import Image
        image=Image.fromarray((rend * 255).astype(np.uint8))
        list_of_images.append(np.array(image))
    
    return rend,list_of_images

def torus_points(image_size=512, num_samples=200, device=None):
    """
    Renders a Torus using parametric sampling. Samples num_samples ** 2 points.
    """

    if device is None:
        device = get_device()

    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta = torch.linspace(0, 2*np.pi, num_samples)
    # Densely sample phi and theta on a grid
    Phi, Theta = torch.meshgrid(phi, theta)
    R=3
    r=2
    x = (R+r*torch.cos(Theta))*torch.cos(Phi)
    y = (R+r*torch.cos(Theta))*torch.sin(Phi)
    z = r*np.sin(Theta)

    points = torch.stack((x.flatten(), y.flatten(), z.flatten()), dim=1)
    color = (points - points.min()) / (points.max() - points.min())
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, 4.0]], device=device,)

    sphere_point_cloud = pytorch3d.structures.Pointclouds(
        points=[points], features=[color],
    ).to(device)


    list_of_images=[]
    azimuth=np.linspace(0,360,100)
    for i in azimuth:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=10, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R,T=T, device=device)
        renderer = get_points_renderer(image_size=image_size, device=device)
        rend = renderer(sphere_point_cloud, lights=lights,cameras=cameras)
        rend=rend[0, ..., :3].cpu().numpy()
        
        
        from PIL import Image
        image=Image.fromarray((rend * 255).astype(np.uint8))
        list_of_images.append(np.array(image))
    return rend,list_of_images

   
def implicit_torus(image_size=512, voxel_size=64, device=None):
    if device is None:
        device = get_device()
    min_value = -10
    max_value = 10
    Rad=4
    rad=2.5
    X, Y, Z = torch.meshgrid([torch.linspace(min_value, max_value, voxel_size)] * 3)
    voxels = (X**2 + Y**2 + Z**2 + Rad**2 - rad**2)**2 - 4 * Rad**2 * (X**2 + Y**2)

    vertices, faces = mcubes.marching_cubes(mcubes.smooth(voxels), isovalue=0)
    vertices = torch.tensor(vertices).float()
    faces = torch.tensor(faces.astype(int))
    # Vertex coordinates are indexed by array position, so we need to
    # renormalize the coordinate system.
    vertices = (vertices / voxel_size) * (max_value - min_value) + min_value
    textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
    textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0)*torch.tensor([100,0,100]))

    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(
        device
    )
    lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -9.0]], device=device,)
    list_of_images=[]
    azimuth=np.linspace(0,360,100)
    for i in azimuth:

        
        renderer = get_mesh_renderer(image_size=image_size, device=device)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=15, elev=0, azim=i)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend=rend[0, ..., :3].detach().cpu().numpy().clip(0, 1)
        
        from PIL import Image
        image=Image.fromarray((rend * 255).astype(np.uint8))
        list_of_images.append(np.array(image))
    return rend,list_of_images
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--render",
        type=str,
        default="rgbd1",
        choices=["rgbd1","rgbd2","rgbdunion","torus","toruspts"],
    )
    parser.add_argument("--output_path", type=str, default="output_images/bridge.jpg")
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=200)
    args = parser.parse_args()
   
    if args.render=="rgbd1":
        dir=load_rgbd_data()
        points,rgb=unproject_depth_image(torch.tensor(dir["rgb1"]),torch.tensor(dir["mask1"]),torch.tensor(dir["depth1"]),dir["cameras1"])
        image,listofimg=render_plant(points,rgb)
        my_images = np.array(listofimg)
        imageio.mimsave('output_images/pt1.gif', my_images, duration=66,loop=0)
        print("Output has been saved!")

    elif args.render=="rgbd2":
        dir=load_rgbd_data()
        points,rgb=unproject_depth_image(torch.tensor(dir["rgb2"]),torch.tensor(dir["mask2"]),torch.tensor(dir["depth2"]),dir["cameras2"])
        image,listofimg=render_plant(points,rgb)
        my_images = np.array(listofimg)
        imageio.mimsave('output_images/pt2.gif', my_images, duration=66,loop=0)
        print("Output has been saved!")

    elif args.render=="rgbdunion":
        dir=load_rgbd_data()
        points1,rgb1=unproject_depth_image(torch.tensor(dir["rgb1"]),torch.tensor(dir["mask1"]),torch.tensor(dir["depth1"]),dir["cameras1"])
        points2,rgb2=unproject_depth_image(torch.tensor(dir["rgb2"]),torch.tensor(dir["mask2"]),torch.tensor(dir["depth2"]),dir["cameras2"])
        image,listofimg=render_plant(torch.cat([points1,points2]),torch.cat([rgb1,rgb2]))
        my_images = np.array(listofimg)
        imageio.mimsave('output_images/ptunion.gif', my_images, duration=66,loop=0)
        print("Output has been saved!")

    elif args.render=="torus":
        image,listofimg = implicit_torus(image_size=args.image_size)
        my_images = np.array(listofimg)
        imageio.mimsave('output_images/torus.gif', my_images, duration=66,loop=0)
        print("Output has been saved!")

    elif args.render=="toruspts":
        image,listofimg = torus_points(image_size=args.image_size,num_samples=200)
        my_images = np.array(listofimg)
        imageio.mimsave('output_images/toruspts.gif', my_images, duration=100,loop=0)
        print("Output has been saved!")
    else:
        raise Exception("Did not understand {}".format(args.render))

    