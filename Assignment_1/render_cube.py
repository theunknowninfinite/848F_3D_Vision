"""
Sample code to render a cow.

Usage:
    python -m starter.render_mesh --image_size 256 --output_path images/cow_render.jpg
"""
import argparse

import matplotlib.pyplot as plt
import pytorch3d
import torch
import numpy as np
from starter.utils import get_device, get_mesh_renderer, load_cow_mesh
import imageio


def render_cow(
    cow_path="data/cow.obj", image_size=256, color=[0.5, 0.3, 0.2], device=None):
    # The device tells us whether we are rendering with GPU or CPU. The rendering will
    # be *much* faster if you have a CUDA-enabled NVIDIA GPU. However, your code will
    # still run fine on a CPU.
    # The default is to run on CPU, so if you do not have a GPU, you do not need to
    # worry about specifying the device in all of these functions.
    if device is None:
        device = get_device()

    # Get the renderer.
    renderer = get_mesh_renderer(image_size=image_size)

    vertices=torch.Tensor([
                [0,0,0],
                [-1,0,0],
                [-1,1,0],
                [0,1,0],
                [0,0,-1],
                [-1,0,-1],
                [-1,1,-1],
                [0,1,-1],
                ])
    faces=torch.Tensor([
            [0,1,3],
            [1,2,3],
            [1,5,2],
            [2,6,5],
            [7,6,5],
            [5,4,7],
            [0,4,7],
            [0,3,7],
            [3,7,2],
            [7,2,6],
            [0,1,5],
            [0,4,5]
            ])
    vertices = vertices.unsqueeze(0)  # (N_v, 3) -> (1, N_v, 3)
    faces = faces.unsqueeze(0)  # (N_f, 3) -> (1, N_f, 3)
    textures = torch.ones_like(vertices)  # (1, N_v, 3)
    textures = textures * torch.tensor(color)  # (1, N_v, 3)
    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures),
    )
    list_of_images=[]
    mesh = mesh.to(device)
    azimuth=np.linspace(0,360,100)
    for i in azimuth:

        R,T=pytorch3d.renderer.cameras.look_at_view_transform(5,45,i)
        # Prepare the camera:
        cameras=pytorch3d.renderer.FoVPerspectiveCameras(
            R=R, T=T, fov=60, device=device)

        # Place a point light in front of the cow.
        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -9]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend.cpu().numpy()[0, ..., :3]  # (B, H, W, 4) -> (H, W, 3)
        from PIL import Image
        image=Image.fromarray((rend * 255).astype(np.uint8))
        list_of_images.append(np.array(image))
  
    return rend,list_of_images


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cow_path", type=str, default="data/cow.obj")
    # parser.add_argument("--output_path", type=str, default="images/cow_render.jpg")
    parser.add_argument("--image_size", type=int, default=512)
    args = parser.parse_args()
    image,listofimg = render_cow(cow_path=args.cow_path, image_size=args.image_size)
    my_images = np.array(listofimg)
    imageio.mimsave('output_images/cube_gif.gif', my_images, duration=25,loop=0)
    print("Output has been saved!")
