import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from PIL import Image as pilmage
import numpy as np
import random
import shutil

from einops import rearrange
from matplotlib.lines import Line2D
import matplotlib
import matplotlib.pyplot as plt
from torch import einsum

from torchvision import transforms
from torchvision.transforms import functional


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def has_set(self, attr):
        return attr in self and self[attr] is not None


def dict_safe_get(dict, attr):
    return dict[attr] if (attr in dict and dict[attr] is not None) else None


class ValueAnnealing():
    def __init__(self, initial_value, final_value, over_steps):
        self.initial_value = initial_value
        self.step = 0
        self.max_step = over_steps
        self.final_value = final_value
        self.piece = (self.initial_value - self.final_value) / self.max_step

    def next(self):
        value = self.initial_value - (self.piece * self.step)
        if self.step < self.max_step:
            self.step += 1
        return value


class ResizeArgs:
    def __init__(self, **kwargs):
        self.align_to = int(kwargs['align_to']) if dict_safe_get(kwargs, 'align_to') else None
        self.max_long_edge = int(kwargs['max_long_edge']) if dict_safe_get(kwargs,
                                                                           'max_long_edge') else None
        self.max_short_edge = int(kwargs['max_short_edge']) if dict_safe_get(kwargs,
                                                                             'max_short_edge') else None

    @staticmethod
    def parse_from_string(config, sep=';'):
        # flex;8;max;512 = Keeps aspect ratio give or take and assures that each side is divisible by 8 and has max len of 512
        align_to = None
        max_long_edge = None
        max_short_edge = None
        args = config.split(sep)
        for i in range(len(args) // 2):
            if args[2 * i] == 'flex':
                align_to = args[2 * i + 1]
            if args[2 * i] == 'max':
                max_long_edge = args[2 * i + 1]
            if args[2 * i] == 'max_short':
                max_short_edge = args[2 * i + 1]
        return ResizeArgs(align_to=align_to, max_long_edge=max_long_edge,
                          max_short_edge=max_short_edge)


class FlexResize():
    def __init__(self, args):
        self.args = args
        # TODO: Implement max_short edge
        self.align_to = args.align_to
        self.max = args.max_long_edge

    def keep_ar_sizes(self, x, max):
        if max is None: return x.height, x.width
        short_w = x.width < x.height
        ar_resized_long = (max / x.height) if short_w else (max / x.width)
        return int(x.height * ar_resized_long), int(x.width * ar_resized_long)

    def __call__(self, x):
        h, w = self.keep_ar_sizes(x, self.max)
        h = h - (h % self.align_to)
        w = w - (w % self.align_to)
        return functional.resize(x, [h, w], antialias=True)


class RandomAdjustHue():
    def __init__(self, ranges):
        self.min = ranges[0]
        self.max = ranges[1]

    def __call__(self, x):
        r = (random.random() * (self.max - self.min)) + self.min
        return functional.adjust_hue(x, r)


class RandomAdjustSaturation():
    def __init__(self, ranges):
        self.min = ranges[0]
        self.max = ranges[1]

    def __call__(self, x):
        r = (random.random() * (self.max - self.min)) + self.min
        return functional.adjust_saturation(x, r)


class RandomAdjustContrast():
    def __init__(self, ranges):
        self.min = ranges[0]
        self.max = ranges[1]

    def __call__(self, x):
        r = (random.random() * (self.max - self.min)) + self.min
        return functional.adjust_contrast(x, r)


class RandomResize():
    def __init__(self, ranges):
        self.min = ranges[0]
        self.max = ranges[1]

    def __call__(self, x):
        r = (random.random() * (self.max - self.min)) + self.min
        return functional.resize(x, int(r), antialias=True)


first = True


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    global first
    if first:
        plt.switch_backend('agg')
        first = False
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)],
               ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    return plt.gcf()


def plot_param_std_mag(named_parameters):
    '''Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_param_std_mag(self.model.named_parameters())" to visualize the magnitudes'''
    global first
    if first:
        plt.switch_backend('agg')
        first = False
    param_std = []
    param_max = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad):
            layers.append(n)
            param_std.append(p.data.std())
            param_max.append(p.data.abs().max())
    plt.bar(np.arange(len(param_max)), param_max, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(param_max)), param_std, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(param_std) + 1, lw=2, color="k")
    plt.xticks(range(0, len(param_std), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(param_std))
    plt.ylim(bottom=-0.001, top=0.2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("param std")
    plt.title("Average Param magnitudes")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-param', 'std-param', 'zero'])
    plt.tight_layout()
    return plt.gcf()


def standard_imagenet_train_transforms(dest_size):
    transform_pil = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.RandomResizedCrop((dest_size[0], dest_size[1]), scale=(0.35, 1.),
                                     ratio=(.99, 1.01)),
        transforms.RandomRotation((-3, 3), resample=PIL.Image.BILINEAR),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        RandomAdjustContrast((0.7, 1.3)),
        RandomAdjustHue((-0.1, 0.1)),
        RandomAdjustSaturation((.7, 1.3))
    ])
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose([transform_pil, transform_tensor])


def standard_imagenet_val_transforms(dest_size):
    transform_pil = transforms.Compose([
        transforms.Lambda(lambda x: x.convert('RGB')),
        transforms.Resize((dest_size[0], dest_size[1]), antialias=True),
    ])
    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transforms.Compose([transform_pil, transform_tensor])


# assume -1 to 1 tensor
def tensor_to_pil(tensor):
    return pilmage.fromarray(((tensor.clamp(-1, 1) + 1.) * 127.5).detach().squeeze(). \
                             permute(1, 2, 0).data.cpu().numpy().astype(np.uint8))


def pil_to_tensor(pil, transform=None, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    if transform is None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    tensor = transform(pil).unsqueeze(0)
    return tensor


def np_to_pil(npy):
    return pilmage.fromarray(npy.astype(np.uint8))


def pil_to_np(pil):
    return np.array(pil)


def tensor_to_np(tensor, cut_dim_to_3=True):
    if len(tensor.shape) == 4:
        if cut_dim_to_3:
            tensor = tensor[0]
        else:
            return tensor.data.cpu().numpy().transpose((0, 2, 3, 1))
    return tensor.data.cpu().numpy().transpose((1, 2, 0))


def tensor_resample(tensor, dst_size, mode='bilinear'):
    return F.interpolate(tensor, dst_size, mode=mode,
                         align_corners=None if mode == 'nearest' else False)


def pil_resize_short_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_short = (trg_size / pil.width) if short_w else (trg_size / pil.height)
    resized = pil.resize((int(pil.width * ar_resized_short), int(pil.height * ar_resized_short)),
                         pilmage.BICUBIC)
    return resized


def pil_resize_long_edge_to(pil, trg_size):
    short_w = pil.width < pil.height
    ar_resized_long = (trg_size / pil.height) if short_w else (trg_size / pil.width)
    resized = pil.resize((int(pil.width * ar_resized_long), int(pil.height * ar_resized_long)),
                         pilmage.BICUBIC)
    return resized


# Imagenet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
class ImageTensorConverter:
    def __init__(self, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], resize=None, is_bgr=False,
                 mul_by=None, unsqueeze=False, device=None, clamp_to_pil=None, drop_alpha=False):
        '''
        mean = Mean value per channel (In RBG even if is_rgb is true)
        std  = Std value per channel (In RBG even if is_rgb is true)
        resize = Same as torchvision.transforms.Resize OR string in the form of 'flex;<alignment>;max;<max_long_edge>' OR ResizeArgs instance
        is_bgr = swap channels when going from image to tensor
        mul_by = denormalize by this value after being cast to [0,1]
        unsqueeze = unsqueeze the resulting tensor in 0th dim and squeeze the tensor when going to PIL
        device = destination device for the tensor
        clamp_to_pil = Tuple-like [min, max] to which the result will be clamped before casting to PIL
        drop_alpha = convert the PIL to 'RGB' before passing it to ToTensor
        '''
        self.mean = mean
        self.std = std
        self.resize = resize
        self.to_tensor_transform = []
        self.inverse_transform = []

        # To Tensor Transform

        # Remove A channel & convert to RGB
        if drop_alpha:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.convert('RGB')))
        # Resize
        if isinstance(resize, ResizeArgs):
            self.to_tensor_transform.append(FlexResize(resize))
        elif isinstance(resize, str):
            if resize.startswith('flex'):
                self.to_tensor_transform.append(FlexResize(ResizeArgs.parse_from_string(resize)))
            else:
                print("Resize arguments unknown.")
        elif resize is not None:
            self.to_tensor_transform.append(transforms.Resize(resize, antialias=True))
        # ToTensor
        self.to_tensor_transform.append(transforms.ToTensor())
        # Denormalize if needed
        if mul_by:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.mul_(mul_by)))
        # Normalization
        self.to_tensor_transform.append(transforms.Normalize(self.mean, self.std))
        # Channel swap Stuff
        if is_bgr:
            self.to_tensor_transform.append(
                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]))
        if unsqueeze:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.unsqueeze(0)))
        if device:
            self.to_tensor_transform.append(transforms.Lambda(lambda x: x.to(device)))

        # To PIL transform

        if unsqueeze:
            self.inverse_transform.append(transforms.Lambda(lambda x: x.squeeze()))
        if is_bgr:
            self.inverse_transform.append(
                transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]))

        self.inverse_transform.append(
            transforms.Lambda(lambda x: x * torch.tensor(self.std).view(-1, 1, 1).to(x.device)))
        self.inverse_transform.append(
            transforms.Lambda(lambda x: x + torch.tensor(self.mean).view(-1, 1, 1).to(x.device)))
        if clamp_to_pil is not None:
            self.inverse_transform.append(
                transforms.Lambda(lambda x: x.clamp(min=clamp_to_pil[0], max=clamp_to_pil[1])))
        if mul_by:
            self.inverse_transform.append(transforms.Lambda(lambda x: x.div_(mul_by)))
        # Cast it to cpu anyway just before PIL
        self.inverse_transform.append(transforms.Lambda(lambda x: x.cpu()))
        self.inverse_transform.append(transforms.ToPILImage())

        self.to_tensor_transform = transforms.Compose(self.to_tensor_transform)
        self.to_pil_transform = transforms.Compose(self.inverse_transform)

    def __call__(self, input):
        if type(input).__name__ == 'Tensor':
            return self.get_pil(input)
        else:
            return self.get_tensor(input)

    def get_tensor(self, input):
        return self.to_tensor_transform(input)

    def get_pil(self, input):
        return self.to_pil_transform(input.clone())  # Fix some latent bugs with a clone.

    def denormalize_tensor(self, x):
        x = x * torch.tensor(self.std).view(-1, 1, 1).to(x.device)
        x = x + torch.tensor(self.mean).view(-1, 1, 1).to(x.device)
        return x


def imagenet_converter(dst_size):
    converter = ImageTensorConverter(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                                     resize=dst_size, drop_alpha=True)
    return converter


def apply_mask_to_np_image(im, mask, mask_range=None, invert=False):
    imf = im.astype(np.float)
    maskf = mask.astype(np.float)
    if mask_range is None:
        maskf += mask.min()
        maskf /= mask.max()
    else:
        maskf += mask_range[0]
        maskf /= mask_range[0] + mask_range[1]
    if invert:
        maskf = 1. - maskf
    # mask is 1 channel but im is not
    if mask.ndim == im.ndim - 1 or mask.shape[-1] == 1:
        maskf = np.stack([maskf] * im.shape[-1], axis=2)
    masked = imf * maskf
    return masked.astype(im.dtype)


def copy_file(src, dst):
    return shutil.copy2(src, dst)


def tensor_to_image(tensor, fname, ext='png'):
    tensor_to_pil(tensor).save(fname if fname.endswith(ext) else f'{fname}.{ext}')
    return fname if fname.endswith(ext) else f'{fname}.{ext}'


def pil_loader(path, conversion='RGB'):
    with open(path, 'rb') as f:
        img = pilmage.open(f)
        return img.convert(conversion)


def is_image(fname):
    fname = fname.lower()
    exts = ['jpg', 'png', 'bmp', 'jpeg', 'tiff']
    ok = any([fname.endswith(ext) for ext in exts])
    return ok


def images_in_directory(dir):
    ls = os.listdir(dir)
    return sorted(list(filter(is_image, ls)))


def subdirectories(dir, ignore_dirs_starting_with_dot=True):
    ls = os.listdir(dir)
    filt_fn = lambda x: os.path.isdir(os.path.join(dir, x))
    if ignore_dirs_starting_with_dot:
        prev = filt_fn
        filt_fn = lambda x: prev(x) and not x.startswith('.')
    return sorted(
        list(filter(
            lambda x: filt_fn(x),
            ls)
        )
    )

class CrossAttentionMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        n, c, h, w = x.shape
        M = rearrange(x, 'n c h w -> n (h w) c')
        N = rearrange(y, 'n c h w -> n (h w) c')
        # print(M.shape, N.shape)
        # G = torch.bmm(M, N.transpose(1, 2))
        G = einsum('b i d, b j d -> b i j', M, N)
        G = G.mean(-1)
        G = rearrange(G, 'n (h w) -> n h w', h=h, w=w)
        # G = torch.bmm(M.cpu(), N.transpose(1, 2).cpu()).cuda()
        G.div_(h * w * c)
        return G

class GramMatrixPatches(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        M = rearrange(x, 'n c h w -> 1 c (n h w)')
        G = torch.bmm(M, M.transpose(1, 2))
        G.div_(n * h * w * c)
        return G

class GramMatrix(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        n, c, h, w = x.shape
        M = x.view(n, c, h * w)
        G = torch.bmm(M, M.transpose(1, 2))
        G.div_(c * h * w)
        return G


class GramMatrixMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()
        self.gm = GramMatrix()

    def forward(self, x, y):
        out = self.loss(self.gm(x), y)
        return out


class ChannelwiseGaussianBlur(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter = None
        self.d2gaussian = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float) / 16.
        self.d2gaussian = self.d2gaussian.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter is None or x.shape[1] != self.cached_filter.shape[0]:
            self.cached_filter = self.d2gaussian.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter, groups=x.shape[1], padding=1)


class ChannelwiseSobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter1 = None
        self.cached_filter2 = None
        self.xsobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float)
        self.ysobel = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float)
        self.xsobel = self.xsobel.unsqueeze(0).unsqueeze(0)
        self.ysobel = self.ysobel.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter1 is None or x.shape[1] != self.cached_filter1.shape[0]:
            self.cached_filter1 = self.xsobel.repeat(x.shape[1], 1, 1, 1).to(x.device)
        if self.cached_filter2 is None or x.shape[1] != self.cached_filter2.shape[0]:
            self.cached_filter2 = self.ysobel.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter1, groups=x.shape[1], padding=1), \
            F.conv2d(x, self.cached_filter2, groups=x.shape[1], padding=1)


class ChannelwiseSobelMagnitude(nn.Module):
    def __init__(self):
        super().__init__()
        self.dir_sobel = ChannelwiseSobel()

    def forward(self, i):
        x, y = self.dir_sobel(i)
        return (x ** 2 + y ** 2) ** 0.5


class ChannelwiseLaplace(nn.Module):
    def __init__(self):
        super().__init__()
        self.cached_filter = None
        self.laplacian = torch.tensor([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=torch.float)
        self.laplacian = self.laplacian.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        if self.cached_filter is None or x.shape[1] != self.cached_filter.shape[0]:
            self.cached_filter = self.laplacian.repeat(x.shape[1], 1, 1, 1).to(x.device)
        return F.conv2d(x, self.cached_filter, groups=x.shape[1], padding=1)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def absorb_bn(conv, bn):
    conv_w_c = conv.weight.data.clone()
    conv_b_c = conv.bias.data.clone()
    invstd = bn.running_var.clone().add_(bn.eps).pow_(-0.5)
    conv_w_c.data.mul_(invstd.view(conv_w_c.shape[0], 1, 1, 1).expand_as(conv_w_c))
    conv_b_c.data.add_(-bn.running_mean).mul_(invstd)
    if bn.affine:
        conv_w_c.mul_(bn.weight.data.view(conv_w_c.shape[0], 1, 1, 1).expand_as(conv_w_c))
        conv_b_c.mul_(bn.weight.data).add_(bn.bias.data)
    new_conv = nn.Conv2d(in_channels=conv.in_channels, out_channels=conv.out_channels,
                         kernel_size=conv.kernel_size,
                         padding=conv.padding, padding_mode=conv.padding_mode,
                         dilation=conv.dilation, stride=conv.stride)
    new_conv.weight.data = conv_w_c
    new_conv.bias.data = conv_b_c
    return new_conv


def calculate_accuracy_and_loss(net, dataloader, device, loss_criterion=nn.CrossEntropyLoss,
                                output_to_pred=lambda x: torch.argmax(x, 1), dtype=None):
    top1_total, top5_total, total, running_loss = 0., 0., 0., 0.

    net.to(device).eval()
    criterion = loss_criterion()
    with torch.no_grad():
        for data in dataloader:
            images, labels = data

            images = images.to(device)
            if dtype:
                images = images.to(dtype)
            labels = labels.to(device)
            outputs = net(images)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            top1, top5 = accuracy(outputs, labels, topk=(1, 5))
            top1_total += top1 / (100. / labels.size(0))
            top5_total += top5 / (100. / labels.size(0))

            total += labels.size(0)
            # predicted = output_to_pred(outputs.data)
            # correct_prediction = (predicted == labels)
            # correct += correct_prediction.sum().item()

    return (top1_total.item() / total) * 100, (top5_total.item() / total) * 100, running_loss / len(
        dataloader)


import functools


def random_order_cartesian_product(*factors):
    amount = functools.reduce(lambda prod, factor: prod * len(factor), factors, 1)
    index_linked_list = [None, None]
    for max_index in reversed(range(amount)):
        index = random.randint(0, max_index)
        index_link = index_linked_list
        while index_link[1] is not None and index_link[1][0] <= index:
            index += 1
            index_link = index_link[1]
        index_link[1] = [index, index_link[1]]
        items = []
        for factor in factors:
            items.append(factor[index % len(factor)])
            index //= len(factor)
        yield items


def randomized_cartesian_product_fast(seq1, seq2):
    x, y = np.meshgrid(seq1, seq2)
    x, y = x.flatten(), y.flatten()
    rndind = np.random.permutation(range(len(x)))
    x, y = x[rndind], y[rndind]
    del rndind
    for i in range(len(x)):
        yield seq1[x[i]], seq2[y[i]]


def randomized_cartesian_product_fast_ub(*sequences):
    product_fields = np.meshgrid(*sequences)
    product_fields = [x.flatten() for x in product_fields]
    rndind = np.random.permutation(range(len(product_fields[0])))
    product_fields = [x[rndind] for x in product_fields]
    del rndind
    for i in range(len(product_fields[0])):
        yield [j[z[i]] for j, z in zip(sequences, product_fields)]


def notebook_progress_bar(progress):
    from IPython.display import clear_output
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait=True)
    text = "Progress: [{0}] {1:.1f}%".format("#" * block + "-" * (bar_length - block),
                                             progress * 100)
    print(text)


def notebook_interactive_text(format, *phs):
    from IPython.display import clear_output

    clear_output(wait=True)
    text = format % tuple(phs)
    print(text)


def torch_clear_gpu_mem():
    import torch
    import gc
    gc.collect()
    torch.cuda.empty_cache()


def guess_model_device(model):
    return next(model.parameters()).device
