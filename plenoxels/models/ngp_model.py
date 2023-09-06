from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import Parameter
import nerfacc

from plenoxels.models.kplane_field_ngp import KPlaneField
from plenoxels.ops.activations import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    VolumetricSampler, RayBundle, RaySamples
)
from plenoxels.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels.utils.timer import CudaTimer


class NGPModel(nn.Module):
    def __init__(self,
                 grid_config: Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                #  # proposal-sampling arguments
                #  num_proposal_iterations: int = 1,
                #  use_same_proposal_network: bool = False,
                #  proposal_net_args_list: List[Dict] = None,
                #  num_proposal_samples: Optional[Tuple[int]] = None,
                #  num_samples: Optional[int] = None,
                #  single_jitter: bool = False,
                #  proposal_warmup: int = 5000,
                #  proposal_update_every: int = 5,
                #  use_proposal_weight_anneal: bool = True,
                #  proposal_weights_anneal_max_num_iters: int = 1000,
                #  proposal_weights_anneal_slope: float = 10.0,
                 # occupancy grid sampling arguments
                 render_step: int = 1024,
                 render_step_size: float = None,
                 grid_resolution: int = 128,
                 grid_levels: int = 4,
                 alpha_thre: float = 0.0, # Threshold for opacity skipping.
                 cone_angle: float = 0.0, # Should be set to 0.0 for blender scenes but 1./256 for real scenes.
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()
        if isinstance(grid_config, str):
            self.config: List[Dict] = eval(grid_config)
        else:
            self.config: List[Dict] = grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.concat_features_across_scales = concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.render_step = render_step
        self.render_step_size = render_step_size
        self.grid_resolution = grid_resolution
        self.grid_levels = grid_levels
        self.alpha_thre = alpha_thre
        self.cone_angle = cone_angle

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)

        self.field = KPlaneField(
            aabb,
            grid_config=self.config,
            concat_features_across_scales=self.concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        # Occupancy Grid.
        self.scene_aabb = Parameter(aabb.flatten(), requires_grad=False)
        if self.render_step_size is None:
            # auto step size: ~1024 samples in the base level grid
            self.render_step_size = ((self.scene_aabb[3:] - self.scene_aabb[:3]) ** 2).sum().sqrt().item() / self.render_step # type: ignore
        self.occupancy_grid = nerfacc.OccGridEstimator(
            roi_aabb=self.scene_aabb,
            resolution=self.grid_resolution,
            levels=self.grid_levels,
        )

        # Sampler
        self.sampler = VolumetricSampler(
            occupancy_grid=self.occupancy_grid,
            density_fn=self.field.density_fn,
        )

    def step_before_iter(self, step):
        self.occupancy_grid.update_every_n_steps(
            step=step,
            occ_eval_fn=lambda x: self.field.density_fn(x) * self.render_step_size,
        )

    def step_after_iter(self, step):
        pass

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor], ray_indices: Optional[torch.Tensor] = None, num_rays: Optional[int] = None):
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            comp_rgb = nerfacc.accumulate_along_rays(
                weights[..., 0], values=rgb, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulated_weight = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            comp_rgb = torch.sum(weights * rgb, dim=-2)
            accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor, ray_indices: Optional[torch.Tensor] = None, num_rays: Optional[int] = None):
        if ray_indices is not None and num_rays is not None:
            eps = 1e-10
            steps = (ray_samples.starts + ray_samples.ends) / 2
            # Necessary for packed samples from volumetric ray sampler
            depth = nerfacc.accumulate_along_rays(
                weights[..., 0], values=steps, ray_indices=ray_indices, n_rays=num_rays
            )
            accumulation = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
            depth = depth / (accumulation + eps)
            depth = torch.clip(depth, steps.min(), steps.max())
        else:
            steps = (ray_samples.starts + ray_samples.ends) / 2
            one_minus_transmittance = torch.sum(weights, dim=-2)
            depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor, ray_indices: Optional[torch.Tensor] = None, num_rays: Optional[int] = None):
        if ray_indices is not None and num_rays is not None:
            # Necessary for packed samples from volumetric ray sampler
            accumulation = nerfacc.accumulate_along_rays(
                weights[..., 0], values=None, ray_indices=ray_indices, n_rays=num_rays
            )
        else:
            accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):
        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions are 3D.
        # ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
        #     ray_bundle, timestamps=timestamps, density_fns=self.density_fns)
        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
                near_plane=near_far[0, 0],
                far_plane=near_far[0, 1],
                render_step_size=self.render_step_size,
                alpha_thre=self.alpha_thre,
                cone_angle=self.cone_angle,
                timestamps=timestamps,
            )

        field_out = self.field(ray_samples.get_positions(), ray_samples.directions, timestamps)
        rgb, density = field_out["rgb"], field_out["density"]

        # weights = ray_samples.get_weights(density)
        # weights_list.append(weights)
        # ray_samples_list.append(ray_samples)

        # accumulation
        num_rays = len(ray_bundle)
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        weights = nerfacc.render_weight_from_density(
            t_starts=ray_samples.starts[..., 0],
            t_ends=ray_samples.ends[..., 0],
            sigmas=density[..., 0],
            packed_info=packed_info,
        )[0]
        weights = weights[..., None]

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color, ray_indices=ray_indices, num_rays=num_rays)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions, ray_indices=ray_indices, num_rays=num_rays)
        accumulation = self.render_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
        }

        # # These use a lot of GPU memory, so we avoid storing them for eval.
        # if self.training:
        #     outputs["weights_list"] = weights_list
        #     outputs["ray_samples_list"] = ray_samples_list
        # for i in range(self.num_proposal_iterations):
        #     outputs[f"prop_depth_{i}"] = self.render_depth(
        #         weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        # pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] # + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] # + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] # + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
        ]