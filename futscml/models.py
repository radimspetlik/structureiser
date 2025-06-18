import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import PIL
from PIL import Image
from torch.autograd import Variable, Function
from torchvision import models, transforms
import os
import math

# A few things from StyleGan2
class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + eps)

class BlurLayer(nn.Module):
    def __init__(self, kernel=None, normalize=True, flip=False, stride=1):
        super().__init__()
        if kernel is None: kernel = [1, 2, 1]
        kernel = torch.tensor(kernel, dtype=torch.float32)
        kernel = kernel[:, None] * kernel[None, :] # make it square
        kernel = kernel[None, None]
        if normalize:
            kernel = kernel / kernel.sum()
        if flip:
            kernel = kernel[:, :, ::-1, ::-1]
        self.register_buffer('kernel', kernel)
        self.stride = stride

    def forward(self, x):
        # expand kernel channels
        kernel = self.kernel.expand(x.size(1), -1, -1, -1)
        x = F.conv2d(x, kernel, stride=self.stride, padding=int((self.kernel.size(2) - 1) / 2), groups=x.size(1) )
        return x

class EqualConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_channel, in_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias: self.bias = nn.Parameter(torch.zeros(out_channel))
        else: self.bias = None

    def forward(self, input):
        out = F.conv2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )

class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        if bias: self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))
        else: self.bias = None

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul if self.bias is not None else None)
        return out

    def __repr__(self):
        return (f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})")

class EqualConvTranspose2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(in_channel, out_channel, kernel_size, kernel_size))
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias: self.bias = nn.Parameter(torch.zeros(out_channel))
        else: self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(input, self.weight * self.scale, bias=self.bias, stride=self.stride, padding=self.padding)
        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )
# End

class IdentityLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
class RecastConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kh, kw, padding=0):
        super().__init__()
        self.fin = channels_in
        self.fout = channels_out
        self.kh = kh
        self.kw = kw
        self.padding = padding
        
    def calculate_output_size(self, x):
        padding = self.padding
        dilation = 1
        stride = 1
        n = x.size(0)
        c = self.fout
        h = (x.size(2) + 2*padding - dilation * (self.kh - 1) - 1) / stride + 1
        w = (x.size(3) + 2*padding - dilation * (self.kw - 1) - 1) / stride + 1
        assert(h == int(h))
        assert(w == int(w))
        return (n, c, int(h), int(w))

    def recast_kernels(self, x):
        batch = x.size(0)
        return x.view((batch*self.fout,self.fin,self.kh,self.kw))# ,self.fout .transpose(1, 2)
        
    def forward(self, x, k, b=None):
        if b is not None:
            assert(b.numel() == self.fout)
        outsize = self.calculate_output_size(x)
        # get kernels
        kernels = self.recast_kernels(k)
        # the groupped way
        # reshape the batch to be (1, n*c, w, h)
        x_b = x.view(1, -1, x.shape[2], x.shape[3])
        # reshape the kernels to be the same
        output = F.conv2d(x_b, kernels, padding=self.padding, groups=x.shape[0], bias=b)
        output = output.view(*outsize)
        return output
    
class ScaledConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding):
        super().__init__()
        self.conv = nn.Parameter(torch.empty((channels_out, channels_in, kernel_size, kernel_size)))
        self.scale = nn.Parameter(torch.empty((channels_out, channels_in)))
        self.padding = padding
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.conv, a=math.sqrt(5))
        init.normal_(self.scale)
        # if self.bias is not None:
        #     fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        #     bound = 1 / math.sqrt(fan_in)
        #     init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # scaling the input channels is the same as scaling the weights
        weight = self.conv * self.scale[:, :, None, None]
        return F.conv2d(x, weight=weight, padding=self.padding)
    
class ScaledConstrainedConv2d(nn.Module):
    def __init__(self, channels_in, channels_out, kernel_size, padding, bias=True, report=False):
        super().__init__()
        # not dependent on number of input channels
        self.conv = nn.Parameter(torch.empty((channels_out, 1, kernel_size, kernel_size)))
        # self.scale = nn.Parameter(torch.empty((channels_out, channels_in)))
        self.scale = nn.Parameter(torch.empty((channels_out, channels_in)))
        self.bias = nn.Parameter(torch.empty((channels_out,))) if bias else None
        self.padding = padding
        self.channels_out = channels_out
        self.channels_in = channels_in
        self.report = report
        self.reset_parameters()

    def reset_parameters(self):
        init.xavier_normal_(self.conv) # kaiming_uniform_ , a=math.sqrt(5)
        # init.normal_(self.scale)
        init.uniform_(self.scale, -1., 1.)
        with torch.no_grad():
            # self.conv.mul_(self.scale[:, :, None, None].mean(1, keepdim=True))
            self.scale.div_((self.scale).sum(1, keepdim=True)) # Sum == 1, on average there is 1 contributions of all input channels
            # print(self.scale.mean(1))
            # print(self.scale)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.conv)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        # scaling the input channels is the same as scaling the weights
        r = self.conv.repeat(1, self.scale.shape[1], 1, 1)
        w = r * self.scale[:, :, None, None]
        result = F.conv2d(x, weight=w, bias=self.bias, padding=self.padding)
        if self.report:
            pass# print(self.conv[0])
            # print(result[0])
        return result
    
    def forward2(self, x):
        b, c_in, h, w = x.shape
        c_out = self.scale.shape[1]
        rechannel = x.view(b*c_in, 1, h, w)
        # skip bias for now
        cnv = F.conv2d(rechannel, weight=self.conv, padding=self.padding) # , groups=b
        rebatch = cnv.view(b, c_in, c_out, h, w) # assuming padding = same
        rebatch = rebatch.view(b, c_in * c_out, h, w)
        #print(rebatch.shape)
        #print(self.scale.view(1, -1, 1, 1).shape)
        scaled = rebatch * self.scale.view(1, -1, 1, 1)
        #print(scaled.shape)
        scaled = scaled.view(b, c_in, c_out, h, w)
        result = scaled.sum(dim=1)
        #print(result.shape)
        return result
    

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view((x.size(0), -1))

class AdditiveNoise(nn.Module):
    def __init__(self, use_noise=True, sigma=0.2):
        super().__init__()
        self.use_noise = use_noise
        self.sigma = sigma

    def forward(self, x):
        if self.use_noise:
            return x + self.sigma * torch.empty_like(x).normal_()
        return x

class SmoothUpsampleLayer(nn.Module):
    def __init__(self, in_filters, out_filters, scale_factor=2, scaling_mode='nearest', bias=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=scaling_mode)
        self.conv = nn.Conv2d(in_channels=in_filters, out_channels=out_filters, bias=bias, padding=1, kernel_size=3)

    def forward(self, x):
        return self.conv(self.upsample(x))

class SmoothUpsampleLayer3D(nn.Module):
    def __init__(self, in_filters, out_filters, depth, depth_padding, scale_factor=2, scaling_mode='nearest', bias=False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=(1, scale_factor, scale_factor), mode=scaling_mode)
        self.conv = nn.Conv3d(in_channels=in_filters, out_channels=out_filters, bias=bias,
                              padding=(depth_padding, 1, 1),
                              kernel_size=(depth, 3, 3),
                              padding_mode='replicate')

    def forward(self, x):
        x = self.upsample(x)
        b, c, d, h, w = x.shape
        x = torch.nn.functional.interpolate(x, (d, h, w), mode='trilinear')
        return self.conv(x)

class DiscriminatorDoubleFiltersPerLayer(nn.Module):
    def __init__(self, num_filters=64, input_channels=3, n_layers=3, norm_layer='instance_norm', use_bias=True):
        super().__init__()
        self.num_filters = num_filters
        self.input_channels = input_channels
        self.use_bias = use_bias

        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        else:
            self.norm_layer = nn.InstanceNorm2d
        self.net = self.make_net(n_layers, self.input_channels, 1, 4, 2, self.use_bias)
    
    def make_net(self, n, flt_in, flt_out=1, k=4, stride=2, bias=True):
        padding = 1
        model = nn.Sequential()

        model.add_module('conv0', self.make_block(flt_in, self.num_filters, k, stride, padding, bias, None, nn.LeakyReLU))

        flt_mult, flt_mult_prev = 1, 1
        # n blocks (essentially n+1 + out real layers)
        for l in range(1, n + 1):
            flt_mult_prev = flt_mult
            flt_mult = min(2**(l), 8)
            model.add_module(f'conv_{l}', self.make_block(self.num_filters * flt_mult_prev, self.num_filters * flt_mult, 
                                                              k, stride if l < n else 1, padding, bias, self.norm_layer, nn.LeakyReLU))        
        model.add_module('conv_out', self.make_block(self.num_filters * flt_mult, 1, k, 1, padding, bias, None, None))
        return model

    def make_block(self, flt_in, flt_out, k, stride, padding, bias, norm, relu):
        m = nn.Sequential()
        m.add_module('conv', nn.Conv2d(flt_in, flt_out, k, stride=stride, padding=padding, bias=bias))
        if norm is not None:
            m.add_module('norm', norm(flt_out))
        if relu is not None:
            m.add_module('relu', relu(0.2, True))
        return m

    def forward(self, x):
        return self.net(x)


class PerceptualVGG19(nn.Module):
    def __init__(self, feature_layers, use_normalization=True, path=None):
        super().__init__()
        if path is not None:
            model = models.vgg19(pretrained=False)
            model.load_state_dict(torch.load(path), strict=False)
        else:
            model = models.vgg19(pretrained=True)
        model.eval()

        self.model = model
        self.feature_layers = feature_layers
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        self.mean_tensor = None
        self.std_tensor = None
        self.use_normalization = use_normalization

        for param in self.parameters():
            param.requires_grad = False

    def normalize(self, x):
        if not self.use_normalization:
            return x
        x = (x + 1) / 2 # pretty sure this should be multiplied by 255.
        return (x - self.mean) / self.std

    def run(self, x):
        features = []
        h = x
        for f in range(max(self.feature_layers) + 1):
            h = self.model.features[f](h)
            if f in self.feature_layers:
                not_normed_features = h.clone().view(h.size(0), -1)
                features.append(not_normed_features)

        return torch.cat(features, dim=1)

    def forward(self, x):
        h = self.normalize(x)
        return self.run(h)


class ImageToImageGenerator_JohnsonFutschik(nn.Module):
    def __init__(self, norm_layer='batch_norm', use_bias=False, resnet_blocks=9, tanh=False,
                 filters=(64, 128, 128, 128, 128, 64), input_channels=3, output_channels=3, append_blocks=None):
        super().__init__()
        assert norm_layer in [None, 'batch_norm', 'instance_norm']
        self.norm_layer = None
        if norm_layer == 'batch_norm':
            self.norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            self.norm_layer = nn.InstanceNorm2d
        self.use_bias = use_bias
        self.resnet_blocks = resnet_blocks
        self.append_blocks = append_blocks

        self.conv0 = self.relu_layer(in_filters=input_channels, out_filters=filters[0],
                                     size=7, stride=1, padding=3, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.conv1 = self.relu_layer(in_filters=filters[0], out_filters=filters[1],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.conv2 = self.relu_layer(in_filters=filters[1], out_filters=filters[2],
                                     size=3, stride=2, padding=1, bias=self.use_bias,
                                     norm_layer=self.norm_layer, nonlinearity=nn.LeakyReLU(.2))

        self.resnets = nn.ModuleList()
        for i in range(self.resnet_blocks):
            self.resnets.append(
                self.resnet_block(in_filters=filters[2], out_filters=filters[2],
                                  size=3, stride=1, padding=1, bias=self.use_bias,
                                  norm_layer=self.norm_layer, nonlinearity=nn.ReLU()))

        self.upconv2 = self.upconv_layer(in_filters=filters[3] + filters[2], out_filters=filters[4],
                                         size=4, stride=2, padding=1, bias=self.use_bias,
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.upconv1 = self.upconv_layer(in_filters=filters[4] + filters[1], out_filters=filters[4],
                                         size=4, stride=2, padding=1, bias=self.use_bias,
                                         norm_layer=self.norm_layer, nonlinearity=nn.ReLU())

        self.conv_11 = nn.Sequential(
            nn.Conv2d(filters[0] + filters[4] + input_channels, filters[5],
                      kernel_size=7, stride=1, padding=3, bias=self.use_bias),
            nn.ReLU()
        )

        self.end_blocks = None
        if self.append_blocks is not None:
            self.end_blocks = nn.Sequential(
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(num_features=filters[5]),
                nn.Conv2d(filters[5], filters[5], kernel_size=3, bias=self.use_bias, padding=1),
                nn.ReLU()
            )

        self.conv_12 = nn.Sequential(nn.Conv2d(filters[5], output_channels, kernel_size=1, stride=1, padding=0, bias=True))
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
                   norm_layer, nonlinearity):
        out = []
        out.append(nn.Conv2d(in_channels=in_filters, out_channels=out_filters,
                             kernel_size=size, stride=stride, padding=padding, bias=bias))
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

    def upconv_layer(self, in_filters, out_filters, size, stride, padding, bias,
                     norm_layer, nonlinearity):
        out = []
        out.append(SmoothUpsampleLayer(in_filters, out_filters))
        if norm_layer:
            out.append(norm_layer(num_features=out_filters))
        if nonlinearity:
            out.append(nonlinearity)
        return nn.Sequential(*out)


class ImageToImageGenerator_Unet(nn.Module):
    def __init__(self, num_filters=8, norm_layer='batch_norm', input_channels=3, output_channels=3, 
                    use_bias=True, skip_connections=[3, 2, 1, 0], tanh=False):
        super().__init__()
        self.num_filters = num_filters
        self.norm_layer = norm_layer
        self.skip = skip_connections
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.tanh = tanh

        if norm_layer == 'batch_norm':
            norm_layer = nn.BatchNorm2d
        elif norm_layer == 'instance_norm':
            norm_layer = nn.InstanceNorm2d
        if norm_layer is None:
            norm_layer = lambda num_features: IdentityLayer()

        self.conv_0 = nn.Sequential(nn.Conv2d(self.input_channels, self.num_filters,
                                              kernel_size=5, stride=4, padding=2, bias=True),
                                    norm_layer(num_features=self.num_filters),
                                    nn.LeakyReLU(negative_slope=.2))

        self.conv_1 = nn.Sequential(nn.Conv2d(self.num_filters, self.num_filters * 2,
                                                kernel_size=5, stride=4, padding=2, bias=True),
                                    norm_layer(num_features=self.num_filters * 2),
                                    nn.LeakyReLU(negative_slope=.2))

        self.conv_2 = nn.Sequential(nn.Conv2d(self.num_filters * 2, self.num_filters * 4,
                                                kernel_size=5, stride=4, padding=2, bias=True),
                                    norm_layer(num_features=self.num_filters * 4),
                                    nn.LeakyReLU(negative_slope=.2))

        self.conv_3 = nn.Sequential(nn.Conv2d(self.num_filters * 4, self.num_filters * 8,
                                                kernel_size=3, stride=2, padding=1, bias=True),
                                    norm_layer(num_features=self.num_filters * 8),
                                    nn.LeakyReLU(negative_slope=.2))

        self.upconv_4 = nn.Sequential(nn.ConvTranspose2d(self.num_filters * 8, self.num_filters * 4,
                                                kernel_size=4, stride=2, padding=1, bias=True),
                                      norm_layer(num_features=self.num_filters * 4),
                                      nn.LeakyReLU(negative_slope=.2))

        self.upconv_5 = nn.Sequential(nn.ConvTranspose2d(self.num_filters * 4 * (2 if 3 in self.skip else 1), self.num_filters * 2, # (8 if 3 in self.skip else 4)
                                                kernel_size=4, stride=4, padding=0, bias=True),
                                      norm_layer(num_features=self.num_filters * 2),
                                      nn.LeakyReLU(negative_slope=.2))

        self.upconv_6 = nn.Sequential(nn.ConvTranspose2d(self.num_filters * 2 * (2 if 2 in self.skip else 1), self.num_filters, # (4 if 2 in self.skip else 2)
                                                kernel_size=4, stride=4, padding=0, bias=True),
                                      norm_layer(num_features=self.num_filters),
                                      nn.LeakyReLU(negative_slope=.2))

        self.upconv_7 = nn.Sequential(nn.ConvTranspose2d(self.num_filters * 1 * (2 if 1 in self.skip else 1), self.output_channels, # (2 if 1 in self.skip else 1)
                                                kernel_size=4, stride=4, padding=0, bias=True),
                                      norm_layer(num_features=3),
                                      nn.LeakyReLU(negative_slope=.2))

        self.conv_8 = nn.Conv2d(self.output_channels + (self.input_channels if 0 in self.skip else 0), output_channels, 
                                    kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        output_0 = self.conv_0(x)
        output_1 = self.conv_1(output_0)
        output_2 = self.conv_2(output_1)
        output_3 = self.conv_3(output_2)
        output = self.upconv_4(output_3)
        print(output.shape, output_2.shape, output_3.shape)
        output = self.upconv_5(torch.cat((output, output_2), dim=1) if 3 in self.skip else output)
        output = self.upconv_6(torch.cat((output, output_1), dim=1) if 2 in self.skip else output)
        output = self.upconv_7(torch.cat((output, output_0), dim=1) if 1 in self.skip else output)
        output = self.conv_8(torch.cat((output, x), dim=1) if 0 in self.skip else output)
        if self.tanh:
            output = F.tanh(output)
        return output

if __name__ == "__main__":
    net = DiscriminatorDoubleFiltersPerLayer(n_layers=3)
    print(net)
    e = torch.empty((20,3,256,256),dtype=torch.float).to('cuda:0')
    z = AdditiveNoise()(e)

    net = ImageToImageGenerator_Unet(skip_connections=[2,0], norm_layer=None)
    net.to('cuda:0')
    o = net(e)
    print(o.shape)
