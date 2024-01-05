import torch
from pytorch3d.renderer import (
    AlphaCompositor,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    HardPhongShader,
)
import pytorch3d
import mcubes
from utils import get_device , get_mesh_renderer, get_points_renderer
import imageio
import matplotlib.pyplot as plt
import numpy as np

def render_vox(voxel,file_name,output_type="vox",vertices=None,faces=None,cam_dist=15,cam_ele=0,cam_azi=0,image_size=256,colors=None,render_360=True):
    voxel_size=32
    min_value = -1
    max_value = 1
    device = get_device()
    voxel = voxel.detach().cpu().squeeze().numpy()
    # voxel = voxel[0].detach().cpu().numpy()
    renderer = get_mesh_renderer(image_size=image_size, device=device)
    list_of_images=[]
    if vertices == None or faces==None:
        vertices, faces = mcubes.marching_cubes(voxel, isovalue=0.5)
        vertices = torch.tensor(vertices).float()
        faces = torch.tensor(faces.astype(int))

        vertices = (vertices / voxel_size) * (max_value - min_value) + min_value

    if colors is not None:
        textures = torch.ones_like(vertices).unsqueeze(0)   # (1, N_v, 3)
        # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0)*torch.tensor([100,0,100]))
        textures = textures * torch.tensor(colors)  # (1, N_v, 3)
        textures=pytorch3d.renderer.TexturesVertex(textures)
    else:
        textures = (vertices - vertices.min()) / (vertices.max() - vertices.min())
        # textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0)*torch.tensor([100,0,100]))
        textures = pytorch3d.renderer.TexturesVertex(vertices.unsqueeze(0))
    mesh = pytorch3d.structures.Meshes([vertices], [faces], textures=textures).to(device)

    if render_360:
        azimuth=np.linspace(0,360,100)
        for i in azimuth:
            lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
            R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_ele, azim=i)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(mesh, cameras=cameras, lights=lights)
            rend=rend[0, ..., :3].cpu().numpy().astype(np.float32)
            image=(np.clip(rend * 255, 0, 255).astype(np.uint8))
            list_of_images.append(image)

        my_images = np.array(list_of_images)
        imageio.mimsave(f'output_images/{output_type}/{file_name}.gif', my_images, duration=66,loop=0)
        print("Saved Gif")

    else:
        lights = pytorch3d.renderer.PointLights(location=[[0, 0.0, -4.0]], device=device)
        R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_ele, azim=cam_azi)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend=rend[0, ..., :3].cpu().numpy().astype(np.float32)
        image=(np.clip(rend * 255, 0, 255).astype(np.uint8))
        plt.imsave(f"{file_name}.jpg", image)
        print("Saved Image")


def render_mesh(mesh_src,file_name,output_type="vox",vertices=None,faces=None,cam_dist=3,cam_ele=0,cam_azi=0,image_size=256,colors=None,render_360=True):
    device = get_device()
    print(device)
    list_of_images=[]
    if vertices == None or faces==None:
        vertices, faces = mesh_src.verts_packed(),mesh_src.faces_packed()
        vertices = vertices.unsqueeze(0)
        faces = faces.unsqueeze(0)
    if colors == None:
        textures = torch.ones_like(mesh_src.verts_list()[0].unsqueeze(0), device = device)
        textures = textures * torch.tensor([0.3, 0.4, 1], device = device)
        # mesh_src.textures=pytorch3d.renderer.TexturesVertex(textures)
        # colors=[0.5, 0.3, 0.2]
        # textures = textures * torch.tensor(colors).to(device)
    else:
        textures = torch.ones_like(vertices)
        textures = textures * torch.tensor(colors).to(device)

    mesh = pytorch3d.structures.Meshes(
        verts=vertices,
        faces=faces,
        textures=pytorch3d.renderer.TexturesVertex(textures))
    mesh = mesh.to(device)
    if render_360:
            azimuth=np.linspace(0,360,100)
            for i in azimuth:
                R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_ele, azim=i)

                renderer = get_mesh_renderer(image_size=image_size)
                cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
                lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)
                rend = renderer(mesh, cameras=cameras, lights=lights)
                rend = rend[0, ..., :3].detach().cpu().numpy()
                image=(np.clip(rend * 255, 0, 255).astype(np.uint8))
                list_of_images.append(image)
            my_images = np.array(list_of_images)
            imageio.mimsave(f'output_images/{output_type}/{file_name}.gif', my_images, duration=66,loop=0)
            print("Saved Gif")
    else:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist, elev=cam_ele, azim=cam_azi)

        renderer = get_mesh_renderer(image_size=image_size)

        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)

        lights = pytorch3d.renderer.PointLights(location=[[0, 0, -3]], device=device)

        rend = renderer(mesh, cameras=cameras, lights=lights)
        rend = rend[0, ..., :3].detach().cpu().numpy()
        plt.imsave(f"{file_name}.jpg", rend)
        print("Saved Image")



def render_point_cloud(pc,file_name,output_type="vox",vertices=None,faces=None,cam_dist=3,cam_ele=0,cam_azi=90,image_size=256,bg_colors=(1, 1, 1),render_360=True):
    device = get_device()
    list_of_images=[]
    renderer = get_points_renderer(
        image_size=image_size, background_color=bg_colors)
    points = pc.detach().cpu().numpy()
    # print(points)
    vertices = torch.tensor(points[0]).to(device).unsqueeze(0)
    device = vertices.device
    rgb = (torch.ones_like(vertices) * torch.tensor([1.0, 0.0, 0.0], device=device))
    point_cloud_str = pytorch3d.structures.Pointclouds(points=vertices,features=rgb)
    if render_360:
        azimuth=np.linspace(0,360,100)
        for i in azimuth:
            R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist,  elev=cam_ele, azim=i)
            cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(point_cloud_str, cameras=cameras)
            rend = rend[0, ..., :3].cpu().numpy()
            rend = rend / rend.max()
            image=(np.clip(rend * 255, 0, 255).astype(np.uint8))
            list_of_images.append(image)
        my_images = np.array(list_of_images)
        imageio.mimsave(f'output_images/{output_type}/{file_name}.gif', my_images, duration=66,loop=0)
        print("Saved Gif")

    else:
        R, T = pytorch3d.renderer.look_at_view_transform(dist=cam_dist,  elev=cam_ele, azim=cam_azi)
        cameras = pytorch3d.renderer.FoVPerspectiveCameras(R=R, T=T, device=device)
        rend = renderer(point_cloud_str, cameras=cameras)
        rend = rend[0, ..., :3].cpu().numpy()
        rend = rend / rend.max()
        plt.imsave(f"{file_name}.jpg", rend)
        print("Saved Image")