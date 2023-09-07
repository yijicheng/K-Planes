import json
import logging as log
import os
from typing import Tuple, Optional, Any, List, Union
from mpi4py import MPI

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_loading import parallel_load_images
from .ray_utils import get_ray_directions, generate_hemispherical_orbit, get_rays
from .intrinsics import Intrinsics


class RodinSubjectDataset(Dataset):
    def __init__(self,
                 datadir,
                 split: str,
                 rootdir: str,
                 savedir: str,
                 num_subjects: int,
                 max_frames: int,
                 subject_batch_size: Optional[int] = None,
                 downsample: float = 1.0,
                 sampling_weights: Optional[torch.Tensor] = None,
                 weights_subsampled: int = 1,
                 ):


        with open("/mnt/blob2/avatar/person_list/2.18_hd.txt", 'r') as f:
            ALL_SUBJECTS = f.read().splitlines()[:num_subjects]

        local_rank = MPI.COMM_WORLD.Get_rank()
        world_size = MPI.COMM_WORLD.Get_size()
        self.all_sujects = ALL_SUBJECTS[local_rank:][::world_size]

        self.datadir = None # dummy
        self.name = None # dummy
        self.split = split
        self.rootdir = rootdir
        self.savedir = savedir
        self.max_frames = max_frames
        self.subject_batch_size = subject_batch_size
        self.downsample = downsample
        self.near_far = [2.0, 6.0]
        self.scene_bbox = get_360_bbox(datadir, is_contracted=False)
        self.is_ndc = False
        self.is_contracted = False

        if num_subjects is not None:
            self.num_samples = num_subjects
            assert self.num_samples == len(self.all_sujects)
        else:
            self.num_samples = None
            raise RuntimeError("Can't figure out num_samples.")
        self.sampling_weights = sampling_weights
        if self.sampling_weights is not None:
            assert len(self.sampling_weights) == self.num_samples, (
                f"Expected {self.num_samples} sampling weights but given {len(self.sampling_weights)}."
            )
        self.sampling_batch_size = 2_000_000  # Increase this?
        if self.num_samples is not None:
            self.use_permutation = self.num_samples < 100_000_000  # 64M is static
        else:
            self.use_permutation = True
        self.perm = None

        log.info(f"RodinSubjectDataset with {self.num_samples} subjects. "
                 f"Loaded {split} set from {rootdir}. "
                 f"Saving to {savedir}. "
                 f"Sampling without replacement={self.use_permutation}.")

    @property
    def img_h(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.height for i in self.intrinsics]
        return self.intrinsics.height

    @property
    def img_w(self) -> Union[int, List[int]]:
        if isinstance(self.intrinsics, list):
            return [i.width for i in self.intrinsics]
        return self.intrinsics.width

    def reset_iter(self):
        if self.sampling_weights is None and self.use_permutation:
            self.perm = torch.randperm(self.num_samples)
        else:
            del self.perm
            self.perm = None

    def get_rand_ids(self, index):
        assert self.subject_batch_size is not None, "Can't get rand_ids for test split"
        if self.sampling_weights is not None:
            raise ValueError("Not Implemented yet")
            batch_size = self.batch_size // (self.weights_subsampled ** 2)
            num_weights = len(self.sampling_weights)
            if num_weights > self.sampling_batch_size:
                # Take a uniform random sample first, then according to the weights
                subset = torch.randint(
                    0, num_weights, size=(self.sampling_batch_size,),
                    dtype=torch.int64, device=self.sampling_weights.device)
                samples = torch.multinomial(
                    input=self.sampling_weights[subset], num_samples=batch_size)
                return subset[samples]
            return torch.multinomial(
                input=self.sampling_weights, num_samples=batch_size)
        else:
            batch_size = self.subject_batch_size
            if self.use_permutation:
                return self.perm[index * batch_size: (index + 1) * batch_size]
            else:
                return torch.randint(0, self.num_samples, size=(batch_size, ))

    def __len__(self):
        if self.split == 'train':
            return (self.num_samples + self.subject_batch_size - 1) // self.subject_batch_size
        else:
            return self.num_samples

    def __getitem__(self, index, return_idxs: bool = False):
        if self.split == 'train':
            index = self.get_rand_ids(index) # subject_index
            
        self.name = self.all_sujects[index]
        self.datadir = os.path.join(self.rootdir, self.name)

        out = {}
        
        # triplane
        triplane_path = os.path.join(self.savedir, self.name + '.npy')
        if os.path.exists(triplane_path):
            with open(triplane_path, 'rb') as f:
                triplane = np.load(f)
                triplane = torch.as_tensor(triplane)
        else:
            triplane = 0.1 * torch.randn((1, 3 * 32 * 512 * 512))

        triplane_save_path = os.path.join(self.savedir, self.name + '.npy')
        out["triplane"] = triplane
        out["triplane_save_path"] = triplane_save_path

        # ray
        if self.split == 'render':
            frame_ids, transform = load_360_frames(self.datadir, 'test', self.max_frames)
            imgs, poses = load_360_images(frame_ids, self.datadir, 'test', self.downsample)
            render_poses = generate_hemispherical_orbit(poses, n_frames=120)
            self.poses = render_poses
            self.intrinsics = load_360_intrinsics(
                transform, img_h=imgs[0].shape[0], img_w=imgs[0].shape[1],
                downsample=self.downsample)
            imgs = None
        else:
            frame_ids, transform = load_360_frames(self.datadir, self.split, self.max_frames)
            imgs, poses = load_360_images(frame_ids, self.datadir, self.split, self.downsample)
            self.intrinsics = load_360_intrinsics(
                transform, img_h=imgs[0].shape[0], img_w=imgs[0].shape[1],
                downsample=self.downsample)
            
        rays_o, rays_d, imgs = create_360_rays(
            imgs, poses, merge_all=False, intrinsics=self.intrinsics) # [N, H*W, ?]

        out["rays_o"] = rays_o
        out["rays_d"] = rays_d
        out["imgs"] = imgs

        # image processing
        pixels = out["imgs"]
        if self.split == 'train':
            bg_color = torch.rand((1, 3), dtype=pixels.dtype, device=pixels.device) # TODO: [max_frames, 3]
        else:
            if pixels is None:
                bg_color = torch.ones((1, 3), dtype=torch.float32, device='cuda:0')
            else:
                bg_color = torch.ones((1, 3), dtype=pixels.dtype, device=pixels.device)
        # Alpha compositing
        if pixels is not None:
            pixels = pixels[..., :3] * pixels[..., 3:] + bg_color * (1.0 - pixels[..., 3:])
        out["imgs"] = pixels
        out["bg_color"] = bg_color
        out["near_fars"] = torch.tensor([self.near_far])
        out["scene_bbox"] = self.scene_bbox.unsqueeze(0)

        if return_idxs:
            return out, index
        return out


def get_360_bbox(datadir, is_contracted=False):
    radius = 1.5
    return torch.tensor([[-radius, -radius, -radius], [radius, radius, radius]])


def create_360_rays(
              imgs: Optional[torch.Tensor],
              poses: torch.Tensor,
              merge_all: bool,
              intrinsics: Intrinsics) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    directions = get_ray_directions(intrinsics, opengl_camera=True)  # [H, W, 3]
    num_frames = poses.shape[0]

    all_rays_o, all_rays_d = [], []
    for i in range(num_frames):
        rays_o, rays_d = get_rays(directions, poses[i], ndc=False, normalize_rd=True)  # h*w, 3
        all_rays_o.append(rays_o)
        all_rays_d.append(rays_d)

    all_rays_o = torch.cat(all_rays_o, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    all_rays_d = torch.cat(all_rays_d, 0).to(dtype=torch.float32)  # [n_frames * h * w, 3]
    if imgs is not None:
        imgs = imgs.view(-1, imgs.shape[-1]).to(dtype=torch.float32)   # [N*H*W, 3/4]
    if not merge_all:
        num_pixels = intrinsics.height * intrinsics.width
        if imgs is not None:
            imgs = imgs.view(num_frames, num_pixels, -1)  # [N, H*W, 3/4]
        all_rays_o = all_rays_o.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
        all_rays_d = all_rays_d.view(num_frames, num_pixels, 3)  # [N, H*W, 3]
    return all_rays_o, all_rays_d, imgs


def load_360_frames(datadir, split, max_frames: int) -> Tuple[Any, Any]:
    with open(os.path.join(datadir, f"metadata_000000.json"), 'r') as f:
        meta = json.load(f)['cameras'][0]
        # frames = meta['frames']

        # Subsample frames
        if split == 'train':
            frame_ids = np.arange(300-max_frames, max_frames)
        elif split == 'test':
            frame_ids = np.arange(max_frames)
        else:
            frame_ids = np.arange(300)
    return frame_ids, meta


def load_360_images(frame_ids, datadir, split, downsample) -> Tuple[torch.Tensor, torch.Tensor]:
    img_poses = parallel_load_images(
        dset_type="rodin_subject",
        tqdm_title=f'Loading {split} data',
        num_images=len(frame_ids),
        frame_ids=frame_ids,
        data_dir=datadir,
        out_h=None,
        out_w=None,
        downsample=downsample,
    )
    imgs, poses = zip(*img_poses)
    imgs = torch.stack(imgs, 0)  # [N, H, W, 3/4]
    poses = torch.stack(poses, 0)  # [N, ????]
    return imgs, poses


def load_360_intrinsics(transform, img_h, img_w, downsample) -> Intrinsics:
    height = img_h
    width = img_w
    # load intrinsics
    if 'focal_length' in transform and 'sensor_width' in transform:
        fl_x = transform['focal_length'] / transform['sensor_width'] * width
        fl_y = transform['focal_length'] / transform['sensor_width'] * height
    else:
        raise RuntimeError('Failed to load focal length, please check the transforms.json!')

    cx = width / 2
    cy = height / 2
    return Intrinsics(height=height, width=width, focal_x=fl_x, focal_y=fl_y, center_x=cx, center_y=cy)
