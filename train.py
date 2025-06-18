import random
import os
from contextlib import suppress as suppress
from argparse import ArgumentParser

import torch
import torch as th
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as F
import torchvision.models as models
import torchvision.io
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as Tfunc
from torchvision.models import VGG19_Weights, VGG19_BN_Weights
from tqdm import tqdm
from omegaconf import OmegaConf
from einops import repeat, rearrange
from controlnet_aux import LineartDetector

from futscml import (
    pil_loader,
    images_in_directory,
    tensor_resample,
    ImageTensorConverter,
    InfiniteDatasetSampler,
    ValueAnnealing,
    TensorboardLogger,
)
from futscml.stopwatch import Stopwatch
from futscml.util import HWC3
from futscml.models import SmoothUpsampleLayer
from futscml.sds import SDSControlNet
from futscml.futscml import GramMatrix, guess_model_device, pil_to_np


class ImageToImageGenerator_JohnsonFutschik(nn.Module):
    def __init__(self, norm_layer='batch_norm', use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 64), input_channels=3, output_channels=3,
                 append_blocks=None, blur_pool=False, conv_padding_mode='replicate',
                 config=None, **kwargs):
        super().__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm']
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.use_bias = use_bias
        self.blur_pool = blur_pool
        self.conv_padding_mode = conv_padding_mode
        self.config = config
        self.use_attention = config['use_attention'] if 'use_attention' in config else False
        self.resnet_blocks = resnet_blocks
        self.append_blocks = append_blocks

        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=7, stride=1, padding=3, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     conv_padding_mode=self.conv_padding_mode)

        self.conv1 = self.relu_layer(in_filters=filters[0], out_filters=filters[1],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     conv_padding_mode=self.conv_padding_mode)

        self.conv2 = self.relu_layer(in_filters=filters[1], out_filters=filters[2],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2),
                                     conv_padding_mode=self.conv_padding_mode)

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[2], out_filters=filters[2],
                                  size=3, stride=1, padding=1, bias=self.use_bias,
                                  norm_layer=self.norm_layer, nonlinearity=nn.ReLU()))

        self.upconv2 = self.upconv_layer(in_filters=filters[3] + filters[2], out_filters=filters[4],
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.upconv1 = self.upconv_layer(in_filters=filters[4] + filters[1], out_filters=filters[4],
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(filters[0] + filters[4] + input_channels, filters[5],
                      kernel_size=7, stride=1, padding=3, bias=self.use_bias, padding_mode=self.conv_padding_mode),
            nn.ReLU()
        )

        # initialize context to gaussian random noise
        self.context = torch.randn(1, 100,
                                   config['attention_context_dim'] if 'attention_context_dim' in config and config[
                                       'attention_context_dim'] is not None else 1)

        self.end_blocks = None
        if self.append_blocks is not None:
            self.end_blocks = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1,
                          padding_mode=self.conv_padding_mode),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]),
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1,
                          padding_mode=self.conv_padding_mode),
                nn.ReLU()
            )

        self.conv_12 = nn.Sequential(
            nn.Conv2d(filters[5], output_channels, kernel_size=1, stride=1, padding=0, bias=True))
        if tanh:
            self.conv_12.add_module('tanh', nn.Tanh())

    def forward(self, x):
        output_0 = self.conv0(x)
        output_1 = self.conv1(output_0)
        output = self.conv2(output_1)
        output_2 = self.conv2(output_1)
        for layer in self.resnets:
            output = layer(output) + output

        output = self.upconv2(torch.cat((output, output_2), dim=1))
        output = self.upconv1(torch.cat((output, output_1), dim=1))
        output = self.conv_11(torch.cat((output, output_0, x), dim=1))
        if self.end_blocks is not None:
            output = self.end_blocks(output)
        output = self.conv_12(output)
        return output

    def relu_layer(self, in_filters, out_filters, size, stride, padding, bias,
                   norm_layer, nonlinearity, conv_padding_mode='replicate'):
        out = []
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias,
                             padding_mode=conv_padding_mode))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)

    def resnet_block(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = []
        if nonlinearity:
            out.append(nonlinearity)
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
        return nn.Sequential(*out)

    def upconv_layer(self, in_filters, out_filters, norm_layer, nonlinearity):
        out = []
        out.append(SmoothUpsampleLayer(in_filters, out_filters))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)


class ImageLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.objective = nn.MSELoss()

    def forward(self, x, y):
        return self.objective(x, y)


class Vgg19_Extractor(nn.Module):
    def __init__(self, capture_layers):
        super().__init__()
        self.vgg_layers = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        # Load the old model if requested
        # self.vgg_layers.load_state_dict(torch.load('/home/futscdav/model_vault/old_vgg_converted_new_transform.pth'))
        self.vgg_layers = self.vgg_layers.features
        self.len_layers = 37  # len(self.vgg_layers)

        for param in self.parameters():
            param.requires_grad = False
        self.capture_layers = capture_layers

    def forward(self, x):
        feat = []
        if -1 in self.capture_layers:
            feat.append(x)
        i = 0
        for mod in self.vgg_layers:
            x = mod(x)
            i += 1
            if i in self.capture_layers:
                feat.append(x)
        return feat


class InnerProductLoss(nn.Module):
    def __init__(self, capture_layers, device):
        super().__init__()
        self.layers = capture_layers
        self.device = device
        self.vgg = Vgg19_Extractor(capture_layers).to(device)
        self.stored_mean = (torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, -1, 1, 1))
        self.stored_std = (torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, -1, 1, 1))
        self.gmm = GramMatrix()
        self.dist = nn.MSELoss()
        self.cache: Dict[float, List[torch.Tensor]] = {0.: [torch.empty((0))]}  # torch.Tensor
        self.attention_layers = []

    def extractor(self, x):
        # remap x to vgg range
        x = (x + 1.) / 2.
        x = x - self.stored_mean
        x = x / self.stored_std
        res = self.vgg(x)
        return res

    def run_scale(self, frame_y, pure_y, cache_y2: bool = True, scale: float = 1.):
        frame_y = F.interpolate(frame_y, scale_factor=float(scale), mode='bilinear', align_corners=False,
                                recompute_scale_factor=False)
        feat_frame_y = self.extractor(frame_y)
        if cache_y2:
            if scale not in self.cache:
                pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
                self.cache[scale] = [self.gmm(l) for idx, l in enumerate(self.extractor(pure_y))]
            gmm_pure_y = self.cache[scale]
        else:
            pure_y = F.interpolate(pure_y, scale_factor=scale, mode='bilinear', align_corners=False)
            feat_pure_y = self.extractor(pure_y)
            gmm_pure_y = [self.gmm(l) for idx, l in enumerate(feat_pure_y)]

        loss = torch.empty((len(feat_frame_y),)).to(frame_y.device)
        for l in range(len(feat_frame_y)):
            gmm_frame_y = self.gmm(feat_frame_y[l])
            if config.use_patches:
                gmm_frame_y = repeat(gmm_frame_y, '1 h w -> c h w', c=gmm_pure_y[l].shape[0])
            assert gmm_pure_y[l].shape[0] == gmm_frame_y.shape[0]
            assert not (gmm_pure_y[l].requires_grad)
            dist = self.dist(gmm_pure_y[l].detach(), gmm_frame_y)
            loss[l] = dist
        return torch.sum(loss)

    def forward(self, frame_y, pure_y, cache_y2: bool = True):
        scale_1_loss = self.run_scale(frame_y, pure_y, cache_y2, scale=1.0)
        # scale_2_loss = self.run_scale(y1, y2, cache_y2, scale=0.5)
        return scale_1_loss


class InferDataset(Dataset):
    def __init__(self, frames_dir, xform):
        self.root = os.path.join(frames_dir)
        self.frames = images_in_directory(self.root)
        self.tensors = []
        self.stems = []
        self.xform = xform
        for frame in self.frames:
            stem, _ = os.path.splitext(frame)
            x = pil_loader(os.path.join(self.root, frame))
            self.tensors.append(self.xform(x))
            self.stems.append(stem)

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.stems[idx], self.tensors[idx]


class NullAugmentations:
    def __init__(self):
        pass

    def __call__(self, *items):
        return items


class ShapeAugmentations:
    def __init__(self):
        self.angle_min = -9
        self.angle_max = 9
        self.hflip_chance = .5

    def rng(self, min, max):
        return random.random() * (max - min) + min

    def __call__(self, *items):
        angle = self.rng(self.angle_min, self.angle_max)
        hflip = self.rng(0, 1) < self.hflip_chance

        def transform(x):
            if hflip:
                x = Tfunc.hflip(x)
            x = Tfunc.rotate(x, angle)
            return x

        augmented_items = []
        for item in items:
            augmented_items.append(transform(item))
        return augmented_items


class ColorAugmentations:
    def __init__(self):
        self.hflip_chance = 0.5
        self.adjust_contrast_min = 0.7
        self.adjust_contrast_max = 1.3
        self.adjust_hue_min = -0.1
        self.adjust_hue_max = 0.1
        self.adjust_saturation_min = 0.7
        self.adjust_saturation_max = 1.3

    def rng(self, min, max):
        return random.random() * (max - min) + min

    def __call__(self, *items):
        cnts = self.rng(self.adjust_contrast_min, self.adjust_contrast_max)
        hue = self.rng(self.adjust_hue_min, self.adjust_hue_max)
        sat = self.rng(self.adjust_saturation_min, self.adjust_saturation_max)

        def transform(x):
            x = Tfunc.adjust_contrast(x, cnts)
            x = Tfunc.adjust_hue(x, hue)
            x = Tfunc.adjust_saturation(x, sat)
            return x

        augmented_items = []
        for item in items:
            augmented_items.append(transform(item))
        return augmented_items


class TrainingDataset(Dataset):
    def __init__(self, frames_dir, keyframe_dir, xform, data_aux, disable_augment=False):
        self.frames_dir = frames_dir
        self.keyframe_dir = keyframe_dir
        self.xform = xform

        keys_in = [f for f in images_in_directory(self.keyframe_dir)]
        keys_out = [f for f in images_in_directory(self.keyframe_dir)]
        self.keypair_files = list(zip(keys_in, keys_out))

        print(f"Found {len(self.keypair_files)} keyframe pairs in {self.keyframe_dir}")

        self.aux_data = data_aux
        self.pairs = []
        self.stems = []
        for keyframe in self.keypair_files:
            key_in, key_out = keyframe
            stem, _ = os.path.splitext(key_in)
            keyframe_in = pil_loader(os.path.join(self.frames_dir, key_in))
            keyframe_out = pil_loader(os.path.join(self.keyframe_dir, key_out))
            self.pairs.append((keyframe_in, keyframe_out))
            self.stems.append(stem)
        self.shape_augment = ShapeAugmentations() if not disable_augment else NullAugmentations()
        self.color_augment = ColorAugmentations()

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        # choose a random sample from the dataset, different on each call, along with the original keyframe

        shaped0, shaped1 = self.shape_augment(self.pairs[idx][0], self.pairs[idx][1])
        colored0, = self.color_augment(shaped0)

        return self.stems[idx], self.xform(colored0), self.xform(shaped1), self.xform(self.pairs[idx][0]), self.xform(
            self.pairs[idx][1])
        # return self.pairs[idx][0], self.pairs[idx][1]


def log_verification_images(config, log, step, model, dataset, transform, additional_image=None):
    with torch.no_grad():
        model.eval()
        idx, example = enumerate(dataset).__next__()
        f = example[1].to(guess_model_device(model))
        if config['model_params']['input_channels'] == 4:
            ones = th.ones_like(f[:, :1], device=device)
            f = th.cat([ones, f], dim=1)
        pred = model(f)
        if additional_image is not None:
            log.log_image('Keyframe', pil_to_np(transform(additional_image[0].data.cpu())), step, format='HWC')
        log.log_image('First Unpaired Frame', pil_to_np(transform(pred[0].data.cpu())), step, format='HWC')
        log.flush()


def log_verification_video(config, log, step, label, model, dataset, transform, shape, max_frames=None, fps=1):
    with torch.no_grad():
        model.eval()
        vid_tensor = torch.empty(1, min(max_frames, len(dataset.dataset)) if max_frames is not None else len(
            dataset.dataset), shape[1], shape[2], shape[3], device=device)
        i = 0
        for _, batch in enumerate(dataset):
            _, b = batch
            if max_frames is not None and i >= max_frames: break
            if len(b.shape) == 3: b = b.unsqueeze(0)
            b = tensor_resample(b.to(device), [shape[2], shape[3]])
            frame = model(b)
            for j in range(frame.shape[0]):
                vid_tensor[:, i, :, :, :] = transform.denormalize_tensor(frame[j:j + 1])
            i = i + 1
        if torch.numel(vid_tensor) > 0:
            # log.log_video(label, vid_tensor.cpu(), step, fps=fps)
            torchvision.io.write_video(log.location() + f"/{step}.mp4",
                                       (vid_tensor[0].cpu().permute((0, 2, 3, 1)) * 255).to(torch.uint8), fps=8)


class ControlProcessor:
    def __init__(self, config, processor):
        self.config = config
        self.processor = processor
        self.control_type = self.config.get('control_type', 'direct')
        self.cache = {}

    def direct_control(self, stem, frame_x_0_255):
        control_image_0_255 = self.processor(rearrange(frame_x_0_255.cpu(), 'c h w -> h w c'), stem=stem,
                                             return_pil=False)
        return th.tensor(rearrange(control_image_0_255, 'h w c -> 1 c h w'), device=frame_x_0_255.device)

    def __call__(self, key_stems, stems, frame_x_m1p1, keyframe_x_m1p1, keyframe_y_m1p1):
        frame_x_0_255 = frame_x_m1p1 * 127.5 + 127.5
        control_images_0_255 = []
        for frame_x_idx in range(frame_x_0_255.shape[0]):
            if self.control_type == 'direct':
                control_images_0_255.append(self.direct_control(stems[frame_x_idx], frame_x_0_255[frame_x_idx]))
            elif self.control_type == 'differentiable':
                control_images_0_255.append(
                    self.differentiable_control(stems[frame_x_idx], frame_x_0_255[frame_x_idx:frame_x_idx + 1]))
            elif self.control_type == 'warped':
                assert len(key_stems) == 1
                cache_key = f'{key_stems[0]}--{stems[frame_x_idx]}'
                if cache_key not in self.cache:
                    self.cache[cache_key] = self.warped_control(key_stems[0], cache_key, frame_x_0_255[frame_x_idx],
                                                                keyframe_x_m1p1, keyframe_y_m1p1)
                control_images_0_255.append(self.cache[cache_key])
            else:
                raise ValueError(f'Unknown control type: {self.control_type}')

        control_image_0_1 = torch.cat(control_images_0_255, dim=0) / 255.0

        return control_image_0_1



class PatchSampler:
    def __init__(self, patch_size: int, num_patches: int):
        """
        Args:
            patch_size: size of the square patch (k)
            num_patches: how many patches to return per call
        """
        self.ps = patch_size
        self.num_patches = num_patches

        # Will be set when we first see an image:
        self._positions = None  # Tensor of shape (total_positions, 2)
        self._ptr = 0

    def _init_perm(self, img: torch.Tensor):
        """
        Given one image tensor of shape (B, C, H, W),
        build & shuffle all valid (y,x) top-left patch coords.
        """
        _, _, H, W = img.shape
        # all valid y and x such that y + ps <= H, x + ps <= W
        ys = torch.arange(0, H - self.ps + 1, device=img.device)
        xs = torch.arange(0, W - self.ps + 1, device=img.device)
        Y, X = torch.meshgrid(ys, xs, indexing='ij')          # both shapes (H-ps+1, W-ps+1)
        coords = torch.stack([Y.flatten(), X.flatten()], dim=1)  # (N, 2)
        N = coords.shape[0]

        # shuffle once
        perm = torch.randperm(N, device=img.device)
        self._positions = coords[perm]  # (N, 2), now a random order
        self._ptr = 0

    def cut_patches(self, images: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Args:
            images: list of tensors, each of shape (B, C, H, W)
        Returns:
            list of tensors of patches; each tensor has shape (B * num_patches, C, ps, ps)
        """
        # (re-)initialize if first call or running out of positions
        if (self._positions is None
            or self._ptr + self.num_patches > self._positions.size(0)):
            self._init_perm(images[0])

        # grab next slice of coords
        coords = self._positions[self._ptr : self._ptr + self.num_patches]
        self._ptr += self.num_patches

        outputs = []
        for img in images:
            # for each (y,x), slice out img[..., y:y+ps, x:x+ps]
            # stack them along a new axis=2 → (B, C, num_patches, ps, ps)
            stacked = torch.stack([
                img[..., y : y + self.ps, x : x + self.ps]
                for (y, x) in coords
            ], dim=2)

            # flatten batch & patch dims → (B * num_patches, C, ps, ps)
            patches = rearrange(stacked, 'b c n k1 k2 -> (b n) c k1 k2')

            outputs.append(patches)

        return outputs


import bisect
from typing import TypeVar, Optional, Dict, List

V = TypeVar("V")


def closest_value(d: Dict[int, V], x: int, width: int) -> Optional[V]:
    """
    Given a dict `d` with integer keys and any values,
    snap the input `x` to the nearest key’s value—unless `x`
    falls within `width` “no-man’s-land” around a midpoint
    between two adjacent keys, in which case return None.
    """
    if not d:
        return None

    # Sort the keys once
    keys = sorted(d)
    n = len(keys)

    # If x is before the first key or after the last, just clamp
    if x <= keys[0]:
        return keys[0]
    if x >= keys[-1]:
        return keys[-1]

    # If x exactly matches a key, return immediately
    if x in d:
        return x

    # Find the place where x would go in the sorted key list
    idx = bisect.bisect_left(keys, x)
    low = keys[idx - 1]
    high = keys[idx]

    # Compute the exact midpoint (float)
    mid = (low + high) / 2

    # Half-range of the “no-man’s-land” window
    half = width // 2

    # If x is too close to the midpoint, we’re in no-man’s-land
    if abs(x - mid) <= half:
        return None

    # Otherwise, snap to the closer of low/high
    target_key = low if x < mid else high
    return target_key


def train(config, model, iters, key_weight, style_weight, structure_weight, dataset_train, dataset_aux,
          dataset_val, transform, device, log):
    model.to(device)

    if key_weight > 0.:
        image_loss = ImageLoss()

    params_to_optimize = list(model.parameters())

    optimizer = opt.AdamW(params_to_optimize, lr=3e-5)  # RMSprop works at 2e-4
    aux_sample = InfiniteDatasetSampler(dataset_aux)
    ebest = float('inf')

    log_image_update_every = config['log_image_update_every'] if 'log_image_update_every' in config else 5000
    log_video_update_every = config['log_video_update_every'] if 'log_video_update_every' in config else 5000

    stopwatch = Stopwatch()
    snapshots = [
        (1 * 5 * 60, '05m'),
        (1 * 15 * 60, '15m'),
        (1 * 30 * 60, '30m'),
        (1 * 60 * 60, '01h'),
        (2 * 60 * 60, '02h'),
        (3 * 60 * 60, '03h'),
        (6 * 60 * 60, '06h'),
        (12 * 60 * 60, '12h')
    ]

    image_error_weight_annealing = ValueAnnealing(key_weight * 5, key_weight / 1024, 20_000)
    trange = tqdm(range(iters))

    control_processor = ControlProcessor(config, processor)

    for epoch in trange:
        # Reset to train mode & init random with new seed
        model.train()
        np.random.seed(epoch)

        with suppress():
            for batch_idx, batch in enumerate(dataset_train):
                error, style_loss, key_loss, structure_loss = 0, 0, 0, 0
                key_stems, *batch = batch
                batch = [thing.to(device) for thing in batch]
                keyframe_x, keyframe_y, pure_x, pure_y = batch

                _, aux_batch = aux_sample()
                stems, frame_x = aux_batch

                control_image_0_1 = control_processor(key_stems, stems, frame_x, keyframe_x, keyframe_y)
                control_image_0_1 = control_image_0_1.to(device)

                frame_x = frame_x.to(device)

                pure_y_full = pure_y.clone()
                if config.use_patches:
                    keyframe_x, keyframe_y, pure_x, pure_y = \
                        sampler.cut_patches([keyframe_x, keyframe_y, pure_x, pure_y])

                optimizer.zero_grad()

                with suppress():
                    y = model(keyframe_x.clone())

                # L1 Loss Calculation
                key_loss += key_weight * image_loss(y, keyframe_y)

                with suppress():
                    frame_y = model(frame_x.clone())

                with suppress():
                    style_loss = style_weight * similarity_loss(frame_y, pure_y_full, cache_y2=True)
                    structure_loss = structure_weight * guidance_sd.train_step(frame_y / 2.0 + 0.5,
                                                                               control_image_0_1,
                                                                               epoch=epoch,
                                                                               inference_step=config['inference_step'])

                # Track values for logging
                error = style_loss + structure_loss + key_loss

                tracked_scalars = ['image_error', 'similarity_error', 'sds_loss', 'error']
                scalars = {name: value for name, value in locals().items() if name in tracked_scalars}
                log.log_multiple_scalars(scalars, epoch)

                error.backward()
                optimizer.step()

                trange.set_postfix({'err': f'{error:0.5f}',
                                    'key': f'{key_loss:0.5f}',
                                    'sty': f'{style_loss:0.5f}',
                                    'str': f'{structure_loss:0.5f}',
                                    })

            # Take snapshots
            for deadline, snap in snapshots:
                if stopwatch.just_passed(deadline):
                    log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()},
                                       f'{snap}_snapshot')
                    log.log_file(log._best_checkpoint_location(), output_name=f'{snap}_snapshot_best.pth')

            if config['max_time_minutes'] is not None and stopwatch.just_passed(config['max_time_minutes'] * 60):
                log_verification_video(config, log, epoch, 'Auxiliary Frames', model, dataset_aux, transform, y.shape,
                                       max_frames=None)
                log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')
                print("Maximum time passed, exiting..")
                return

            if epoch % log_image_update_every == 0 and epoch != 0:
                log_verification_images(config, log, epoch, model, dataset_aux, transform, y)

                if error < ebest:
                    ebest = error
                    log.log_checkpoint_best({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()})

            if epoch % log_video_update_every == 0 and epoch != 0:
                if dataset_val is not None:
                    log_verification_video(config, log, epoch, 'Validation Frames', model, dataset_val, transform,
                                           frame_x.shape, max_frames=500, fps=25)
                log_verification_video(config, log, epoch, 'Auxiliary Frames', model, dataset_aux, transform, frame_x.shape,
                                       max_frames=None)
                log.flush()
                log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')
    log.log_checkpoint({'state_dict': model.state_dict(), 'opt_dict': optimizer.state_dict()}, 'latest')


class CachedControlProcessor:
    def __init__(self):
        self.cache = {}

    def call(self, input_image, *_, **__):
        raise NotImplementedError

    def __call__(self, frame, stem=None, *args, **kwargs):
        if stem is not None and stem in self.cache:
            return self.cache[stem]

        if not isinstance(frame, np.ndarray):
            frame = np.array(frame, dtype=np.uint8)
        frame = HWC3(frame)

        control_image = self.call(frame, *args, **kwargs)

        self.cache[stem] = control_image
        return self.cache[stem]


class CacheControlProcessor(CachedControlProcessor):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def __call__(self, frame, stem=None, *args, **kwargs):
        if stem is not None and stem in self.cache:
            return self.cache[stem]

        self.cache[stem] = self.processor(frame, *args, **kwargs)
        return self.cache[stem]


def prepare_cldm(config):
    if config['cldm_type'] == 'lineart':
        guidance_sd = SDSControlNet(device, fp16=False)
        processor = CacheControlProcessor(LineartDetector.from_pretrained("lllyasviel/Annotators"))
    else:
        raise ValueError(f"Unknown CLDM type {config['cldm_type']}")
    guidance_sd.get_text_embeds([config['prompt'] if config['prompt'] is not None else ""],
                                [config['negative_prompt'] if config['negative_prompt'] is not None else ""])

    return guidance_sd, processor


class ModelMock:
    def __init__(self):
        pass

    def forward(self, x):
        return x

    def to(self, device):
        pass

    def eval(self):
        pass

    def train(self):
        pass

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def parameters(self):
        return []

    def zero_grad(self):
        pass

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument('config_file')
    adrgs = parser.parse_args()

    with open(adrgs.config_file, 'r') as f:
        config = OmegaConf.load(f)

    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    np.random.seed(0)

    key_frames_dir = config['key_frames_dir']
    frames_dir = config['frames_dir']

    # Additional validation data if valid exists
    data_root_valid = None
    if os.path.exists(os.path.join(config['key_frames_dir'], 'valid')):
        data_root_valid = os.path.join(config['key_frames_dir'], 'valid')

    # probe size
    data_aux_probe = InferDataset(frames_dir, lambda x: x)
    data_train_probe = TrainingDataset(frames_dir, key_frames_dir, lambda x: x, data_aux_probe,
                                       disable_augment=config['disable_augment'])

    size = None
    for pair in data_train_probe.pairs:
        x, y = pair
        if size is None: size = x.size
        if x.size != size:
            print("WARNING: One of the input images has different size.")
        if y.size != size:
            print("WARNING: One of the output images has different size")
    for im in data_aux_probe.tensors:
        if im.size != size:
            print("WARNING: One of the video frames has different size")

    del data_aux_probe
    del data_train_probe

    device = config['device']
    storage_to_cpu = False
    transform = ImageTensorConverter(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5],
                                     resize=f'flex;8;max;{config["resize"]}' if config["resize"] is not None else f'flex;8',
                                     drop_alpha=True)
    model = ImageToImageGenerator_JohnsonFutschik(config=config, **config['model_params'])

    data_aux = InferDataset(frames_dir, transform)
    data_train = TrainingDataset(frames_dir, key_frames_dir, transform, data_aux,
                                 disable_augment=config['disable_augment'])
    data_validate = InferDataset(data_root_valid, transform) if data_root_valid is not None else None


    def worker_init_fn(worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id)

    batch_size = config['batch_size']
    collate = lambda x: [torch.utils.data.dataloader.default_collate(x).to(device) for x in x]
    trainset = DataLoader(data_train, num_workers=0, worker_init_fn=worker_init_fn)
    auxset = DataLoader(data_aux, num_workers=0, worker_init_fn=worker_init_fn, batch_size=batch_size, drop_last=False)
    testset = DataLoader(data_validate, num_workers=0) if data_validate is not None else None

    key_weight = config['key_weight']
    style_weight = config['style_weight']
    structure_weight = config['structure_weight']

    log = TensorboardLogger(config['logdir'], checkpoint_fmt='checkpoint_%s.pth')
    with open(os.path.join(log.location(), log.experiment_name() + '.yml'), 'w') as f:
        OmegaConf.save(config, f)

    layers = config['vgg_layers']

    sampler = PatchSampler(config.patch_size, config.num_patches)

    similarity_loss = InnerProductLoss(layers, device)

    guidance_sd, processor = prepare_cldm(config)

    train(config, model, config['iters'], key_weight, style_weight, structure_weight,
          trainset, auxset, testset,
          transform, device, log)

