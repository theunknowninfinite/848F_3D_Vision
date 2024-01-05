import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase

if torch.cuda.is_available():
    print("Using CUDA")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(self.min_depth,self.max_depth,self.n_pts_per_ray).view(1, -1, 1).to(device)
        # TODO (1.4): Sample points from z values
        sample_points = ray_bundle.origins.view(-1, 1, 3) + z_vals * ray_bundle.directions.view(-1, 1, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points.to(device),
            sample_lengths=z_vals * torch.ones_like(sample_points[..., :1]),
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}