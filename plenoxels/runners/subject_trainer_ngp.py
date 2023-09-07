import math
import os
from copy import copy
import logging as log
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Sequence, Any, Optional

import pandas as pd
import numpy as np
import torch
import torch.utils.data
from torch.utils.data import Dataset

from plenoxels.datasets import SyntheticNerfDataset, LLFFDataset, RodinSubjectDataset
from plenoxels.models.ngp_model_subject import NGPModel
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.utils.parse_args import parse_optint
from .ddp_trainer_ngp import BaseTrainer, init_dloader_random
from .regularization import (
    PlaneTV, HistogramLoss, L1ProposalNetwork, DepthTV, DistortionLoss,
)
from plenoxels.ops.lr_scheduling import (
    get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
)

from plenoxels.utils.dist_util import is_main_process
from plenoxels.datasets.base_dataset import BaseDataset
class PixelSampler:
    def __init__(self, total, batch):
        self.total = total
        self.batch = batch
        self.curr = total
        self.ids = None

    def nextids(self):
        self.curr+=self.batch
        if self.curr + self.batch > self.total:
            self.ids = torch.LongTensor(np.random.permutation(self.total))
            self.curr = 0
        return self.ids[self.curr:self.curr+self.batch]

# class RayDataset(BaseDataset):
#     def __init__(self, data_subject, ray_batch_size, split):
#         super().__init__(
#             datadir=os.path.dirname(data_subject["triplane_save_path"]),
#             split=split,
#             scene_bbox=data_subject["scene_box"],
#             is_ndc=False,
#             is_contracted=False,
#             batch_size=ray_batch_size,
#             imgs=data_subject["imgs"],
#             rays_o=data_subject["rays_o"],
#             rays_d=data_subject["rays_d"],
#             intrinsics=None,
#         )

#     def __getitem__(self, index):
#         out = super().__getitem__(index)

#         return out

    
class StaticTrainer(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.TensorDataset,
                 tr_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.test_dataset = ts_dset
        self.train_dataset = tr_dset
        self.is_ndc = self.test_dataset.is_ndc
        self.is_contracted = self.test_dataset.is_contracted
        self.num_epochs = kwargs['num_epochs']
        self.batch_size = kwargs['batch_size']
        self.finetune_mlp = kwargs['finetune_mlp']

        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            **kwargs
        )

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            # near_far and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, near_far=near_far,
                                     bg_color=bg_color)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}

    def train_step(self, data, **kwargs) -> bool:
        self.model.train()
        data = self._move_data_to_device(data)
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(
                data['rays_o'], data['rays_d'], bg_color=data['bg_color'],
                near_far=data['near_fars'], timestamps=data['timestamps'])
            self.timer.check("model-forward")
            # Reconstruction loss
            recon_loss = self.criterion(fwd_out['rgb'], data['imgs'])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss
            self.timer.check("regularizaion-forward")
        # Update weights
        self.optimizer_triplane.zero_grad(set_to_none=True)
        if self.finetune_mlp:
            self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.timer.check("backward")
        self.gscaler.step(self.optimizer_triplane)
        if self.finetune_mlp:
            self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        self.train_dataset.reset_iter()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        for epoch in self.num_epochs:
            log.info(f"Starting training from step {self.global_step + 1}, epoch {epoch}")
            
            pb_epoch = tqdm(initial=0, total=len(self.train_dataset), desc=f"Epoch {epoch}: ")
            try:
                self.pre_epoch()
                subject_iter = iter(self.train_data_loader)

                for _ in range(len(self.train_dataset)):
                    self.timer.reset()
                    data_subject = next(subject_iter)
                    self.timer.check("subject-loader-next")

                    train_sampler = PixelSampler(data_subject["rays_o"].shape[1], self.batch_size)
            
                    self.latent = data_subject["triplane"]
                    self.latent.requires_grad = True
                    self.triplane_save_dir = data_subject["triplane_save_path"]
                    self.optimizer_triplane = self.init_optim_triplane()
                    self.scheduler_triplane = self.init_lr_scheduler_triplane()
                    self.timer.check("subject-loop-init")
 

                    for step_ray in range(self.num_steps):
                        self.timer.reset()
                        # self.model.step_before_iter(step_ray) # must after latent loaded
                        self.global_step += 1
                        data_ray = {}
                        indexes_ray = train_sampler.nextids()
                        data_ray["rays_o"] = data_subject["rays_o"][:, indexes_ray, :]
                        data_ray["rays_d"] = data_subject["rays_d"][:, indexes_ray, :]
                        data_ray["imgs"] = data_subject["imgs"][:, indexes_ray, :]
                        data_ray["near_fars"] = data_subject["near_fars"]
                        data_ray["bg_color"] = data_subject["bg_color"]
                        data_ray["latent"] = self.latent
                        data_ray["step_ray"] = step_ray
                        self.timer.check("ray-loader-next")
                        step_successful = self.train_step(data_ray)

                        if step_successful and self.scheduler is not None:
                            self.scheduler_triplane.step()
                            if self.finetune_mlp:
                                self.scheduler.step()
                        for r in self.regularizers:
                            r.step(self.global_step)
                        self.post_step(progress_bar=pb_epoch)
                        self.timer.check("after-ray-loop")
            finally:
                pb_epoch.close()
            self.writer.close()

    def _move_data_to_device(self, data):
        super()._move_data_to_device(data)
        data["latent"] = data["latent"].to(self.device)

    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        for img_idx, data in enumerate(dataset):
            ts_render = self.eval_step(data)
            out_metrics, _, _ = self.evaluate_metrics(
                data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                name=None, save_outputs=self.save_outputs)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name="")
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))

    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def save_model(self):
        model_fname = os.path.join(self.log_dir, f'model.pth')
        log.info(f'Saving model checkpoint to: {model_fname}')
        if is_main_process:
            torch.save(self.get_save_dict(), model_fname)
        torch.save(self.latent.detach().cpu().numpy(), self.triplane_save_dir)

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler_triplane(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type_triplane']
        log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched

    def init_optim_triplane(self, **kwargs) -> torch.optim.Optimizer:
        optim_type = kwargs['optim_type_triplane']
        if optim_type == 'adam':
            optim = torch.optim.Adam(params=[{"params": [self.latent], "lr": kwargs["lr_triplane"]}], eps=1e-15)
        else:
            raise NotImplementedError()
        return optim


    def init_model(self, **kwargs) -> NGPModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            L1ProposalNetwork(kwargs.get('l1_proposal_net_weight', 0.0)),
            DepthTV(kwargs.get('depth_tv_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5


def decide_dset_type(dd) -> str:
    if ("chair" in dd or "drums" in dd or "ficus" in dd or "hotdog" in dd
            or "lego" in dd or "materials" in dd or "mic" in dd
            or "ship" in dd):
        return "synthetic"
    elif ("fern" in dd or "flower" in dd or "fortress" in dd
          or "horns" in dd or "leaves" in dd or "orchids" in dd
          or "room" in dd or "trex" in dd):
        return "llff"
    elif ("render" in dd and not "subject" in dd):
        return "rodin"
    else:
        raise RuntimeError(f"data_dir {dd} not recognized as LLFF or Synthetic dataset.")


def init_tr_data(data_downsample: float, data_dirs: Sequence[str], **kwargs):
    batch_size = int(kwargs['batch_size'])
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]

    dset_type = decide_dset_type(data_dir)
    if dset_type == "synthetic":
        max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
        dset = SyntheticNerfDataset(
            data_dir, split='train', downsample=data_downsample,
            max_frames=max_tr_frames, batch_size=batch_size)
    elif dset_type == "llff":
        hold_every = parse_optint(kwargs.get('hold_every'))
        dset = LLFFDataset(
            data_dir, split='train', downsample=int(data_downsample), hold_every=hold_every,
            batch_size=batch_size, contraction=kwargs['contract'], ndc=kwargs['ndc'],
            ndc_far=float(kwargs['ndc_far']), near_scaling=float(kwargs['near_scaling']))
    elif dset_type == "rodin_subject":
        max_tr_frames = parse_optint(kwargs.get('max_tr_frames'))
        dset = RodinSubjectDataset(
            data_dir, split='train', savedir=kwargs['savedir'], downsample=data_downsample,
            max_frames=max_tr_frames, subject_batch_size=1, num_subjects=kwargs['num_subjects'])
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    dset.reset_iter()

    tr_loader = torch.utils.data.DataLoader(
        dset, num_workers=0, prefetch_factor=None, pin_memory=True,
        batch_size=None, worker_init_fn=init_dloader_random)

    return {
        "tr_dset": dset,
        "tr_loader": tr_loader,
    }


def init_ts_data(data_dirs: Sequence[str], split: str, **kwargs):
    assert len(data_dirs) == 1
    data_dir = data_dirs[0]
    dset_type = decide_dset_type(data_dir)
    if dset_type == "synthetic":
        max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
        dset = SyntheticNerfDataset(
            data_dir, split=split, downsample=1, max_frames=max_ts_frames)
    elif dset_type == "llff":
        hold_every = parse_optint(kwargs.get('hold_every'))
        dset = LLFFDataset(
            data_dir, split=split, downsample=4, hold_every=hold_every,
            contraction=kwargs['contract'], ndc=kwargs['ndc'],
            ndc_far=float(kwargs['ndc_far']), near_scaling=float(kwargs['near_scaling']))
    elif dset_type == "rodin_subject":
        max_ts_frames = parse_optint(kwargs.get('max_ts_frames'))
        dset = RodinSubjectDataset(
            data_dir, split=split, savedir=kwargs['savedir'],
            downsample=1, max_frames=max_ts_frames, 
            subject_batch_size=1, num_subjects=kwargs['num_subjects'])
    else:
        raise ValueError(f"Dataset type {dset_type} invalid.")
    return {"ts_dset": dset}


def load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    od: Dict[str, Any] = {}
    if not validate_only:
        od.update(init_tr_data(data_downsample, data_dirs, **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs, split=test_split, **kwargs))
    return od

def initialize_model(
        runner: Union['StaticTrainer', 'PhototourismTrainer', 'VideoTrainer'],
        **kwargs) -> NGPModel:
    """Initialize a `NGPModel` according to the **kwargs parameters.

    Args:
        runner: The runner object which will hold the model.
                Needed here to fetch dataset parameters.
        **kwargs: Extra parameters to pass to the model

    Returns:
        Initialized NGPModel.
    """
    from .phototourism_trainer import PhototourismTrainer
    extra_args = copy(kwargs)
    extra_args.pop('global_scale', None)
    extra_args.pop('global_translation', None)

    dset = runner.test_dataset
    try:
        global_translation = dset.global_translation
    except AttributeError:
        global_translation = None
    try:
        global_scale = dset.global_scale
    except AttributeError:
        global_scale = None

    num_images = None
    if runner.train_dataset is not None:
        try:
            num_images = runner.train_dataset.num_images
        except AttributeError:
            num_images = None
    else:
        try:
            num_images = runner.test_dataset.num_images
        except AttributeError:
            num_images = None
    model = NGPModel(
        grid_config=extra_args.pop("grid_config"),
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        use_appearance_embedding=isinstance(runner, PhototourismTrainer),
        num_images=num_images,
        **extra_args)
    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.linear_decoder}.")
    return model
