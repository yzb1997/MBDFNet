import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.archs.vgg_arch import VGGFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss
from torch.distributions.laplace import Laplace
from basicsr.utils.img_util import gau_pyramid, imgtoDivisible
import torch.optim as optim
from einops import rearrange
from basicsr.utils import img2tensor, tensor2img
from pytorch_wavelets import DWTForward
import itertools
from math import exp

import cv2

from basicsr.metrics.metric_util import reorder_image, to_y_channel

import numpy as np

_reduction_modes = ['none', 'mean', 'sum']

laplace = Laplace(torch.Tensor([0.0]), torch.Tensor([1.0]))

IS_HIGH_VERSION = tuple(map(int, torch.__version__.split('+')[0].split('.'))) > (1, 6, 1)
if IS_HIGH_VERSION:
    import torch.fft


@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)



########################################################################################################################
'''
    losses of Deep-RFT
    CharbonnierLoss
    EdgeLoss
    fftLoss
'''
@LOSS_REGISTRY.register()
class CharbonnierrftLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, loss_weight=1.0, eps=1e-6):
        super(CharbonnierrftLoss, self).__init__()
        self.loss_weight = loss_weight
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.mean(torch.sqrt((diff * diff) + (self.eps*self.eps)))
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(EdgeLoss, self).__init__()
        self.loss_weight = loss_weight
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()
        if torch.cuda.is_available():
            self.kernel = self.kernel
        self.loss = CharbonnierLoss()

    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = self.loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss * self.loss_weight

@LOSS_REGISTRY.register()
class fftLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        self.loss_weight = loss_weight
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        if IS_HIGH_VERSION:
            x_fft = torch.fft.rfft(x)
            x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)

            target_fft = torch.fft.rfft(y)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        else:
            x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
            target_fft = torch.rfft(y, signal_ndim=2, normalized=False, onesided=True)

        diff = x_fft - target_fft
        loss = torch.mean(abs(diff))
        return loss * self.loss_weight
########################################################################################################################
@LOSS_REGISTRY.register()
class CTLoss(nn.Module):
    def __init__(self, loss_weight=1.0, reduction='mean', dim=64, wavelet_decomp_level=3):
        super().__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.xfm = DWTForward(J=wavelet_decomp_level, mode='zero', wave='haar')
        self.net = nn.Sequential(
            nn.Conv2d(3, dim*2, 1, 1, 0),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim*2, dim, 1, 1, 0)
        )

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, x1, x2, weight=None, **kwargs):
        """
        x1: the SR image
        x2: the corresponding GT
        """
        x1_l, x1_h = self.xfm(x1)
        x2_l, x2_h = self.xfm(x2)

        # x1_l, x2_l = self.net(x1_l), self.net(x2_l)
        # neg = l1_loss(x2_l, x1_l, weight, reduction=self.reduction)

        pos, neg = 0., 0.
        for i in range(len(x2_h)):
            h2 = self.net(rearrange(x2_h[i], 'b c t h w -> (b t) c h w'))
            h2 = rearrange(h2, '(b t) c h w -> b c t h w', b=x1.size(0))

            h1 = self.net(rearrange(x1_h[i], 'b c t h w -> (b t) c h w'))
            h1 = rearrange(h1, '(b t) c h w -> b c t h w', b=x1.size(0))

            pos += l1_loss(h2, h1, weight, reduction=self.reduction)

            for j, k in itertools.product(range(3), range(3)):
                if j != k:
                    neg += l1_loss(h2[:, :, j], h1[:, :, k], weight, reduction=self.reduction)

        return self.loss_weight * (pos / (neg + 1e-7))

@LOSS_REGISTRY.register()
class CTV2Loss(nn.Module):
    def __init__(self,
                layer_weights=None,
                vgg_type='vgg19',
                use_input_norm=True,
                range_norm=False,
                loss_weight = 1.0,
                criterion='l1',
                wavelet_decomp_level=1):
        super().__init__()
        if layer_weights is None:
            layer_weights = {'conv1_2': 1./32, 'conv2_2': 1./16, 'conv3_4': 1./8}

        self.xfm = DWTForward(J=wavelet_decomp_level, mode='zero', wave='haar')

        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

        self.layer_weights = layer_weights
        self.loss_weight = loss_weight

    def forward(self, x1, x2):
        """
        x1: the SR tensor with shape (b, c, h, w)
        x2: the corresponding GT tensor with shape (b, c, h, w)
        """
        _, x1_h = self.xfm(x1)
        _, x2_h = self.xfm(x2)
        # [b, c, 3, h, w], 3 means the directional frequency
        x1_h = self.vgg(rearrange(x1_h[0], 'b c j h w -> (b j) c h w'))
        x2_h = self.vgg(rearrange(x2_h[0], 'b c j h w -> (b j) c h w'))

        loss, pos, neg = 0., 0., 0.
        for k in x1_h.keys():
            pos += self.criterion(x2_h[k], x1_h[k])

            x1_h[k] = rearrange(x1_h[k], '(b j) c h w -> b c j h w', b=x1.size(0))
            x2_h[k] = rearrange(x2_h[k], '(b j) c h w -> b c j h w', b=x1.size(0))
            for i, j in itertools.product(range(3), range(3)):
                if i != j:
                    neg += self.criterion(x2_h[k][:, :, i], x2_h[k][:, :, j])

            loss += self.layer_weights[k] * (pos / (neg + 1e-7))
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class HEMLoss(nn.Module):
    def __init__(self, loss_weight=1.0, hard_thre_p=0.5, device='cuda', random_thre_p=0.1):
        super(HEMLoss, self).__init__()
        self.hard_thre_p = hard_thre_p
        self.loss_weight = loss_weight
        self.random_thre_p = random_thre_p
        self.L1_loss = nn.L1Loss()
        self.device = device

    def hard_mining_mask(self, x, y):
        with torch.no_grad():
            b, c, h, w = x.size()

            hard_mask = np.zeros(shape=(b, 1, h, w)) #创建一个和输入输出一样大的mask，全部都是0
            res = torch.sum(torch.abs(x - y), dim=1, keepdim=True) #将输入和gt相减，然后取绝对值，在channel上相加，之后保存channel维度
            res_numpy = res.cpu().numpy()
            res_line = res.view(b, -1)
            res_sort = [res_line[i].sort(descending=True) for i in range(b)] #对b中的每个列表排序
            hard_thre_ind = int(self.hard_thre_p * w * h)
            for i in range(b):
                thre_res = res_sort[i][0][hard_thre_ind].item()
                hard_mask[i] = (res_numpy[i] > thre_res).astype(np.float32)

            random_thre_ind = int(self.random_thre_p * w * h)
            random_mask = np.zeros(shape=(b, 1 * h * w))
            for i in range(b):
                random_mask[i, :random_thre_ind] = 1.
                np.random.shuffle(random_mask[i])
            random_mask = np.reshape(random_mask, (b, 1, h, w))

            mask = hard_mask + random_mask
            mask = (mask > 0.).astype(np.float32)

            mask = torch.Tensor(mask).to(self.device)

        return mask

    def forward(self, x, y):
        mask = self.hard_mining_mask(x.detach(), y.detach()).detach()

        hem_loss = self.L1_loss(x * mask, y * mask)

        return self.loss_weight * hem_loss

@LOSS_REGISTRY.register()
class BLURDETECTIONLoss(nn.Module):
    def __init__(self, loss_weight=1.0, device='cuda'):
        super(BLURDETECTIONLoss, self).__init__()
        self.loss_weight = loss_weight
        self.L1_loss = nn.L1Loss()
        self.device = device

        self.detection_net = Binary_2channel_Classification().cuda()
        self.detection_net.load_state_dict(torch.load(
            '../experiments/blurdetection/moreblur_200_epoch_model.pt'),
            strict=False,
        )
        for p in self.detection_net.parameters():
            p.requires_grad=False


    def forward(self, blur, gt, input):
        mask = self.detection_net(input).to(self.device)

        bd_loss = self.L1_loss(blur * mask, gt * mask)

        return self.loss_weight * bd_loss

@LOSS_REGISTRY.register()
class TVLoss(nn.Module):
    """Total Variation Loss"""
    def __init__(self, loss_weight=1.0):
        super(TVLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x):
        return self.loss_weight * (torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])) + \
               torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])))

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, window=None, size_average=True, full=False, val_range=None):
    # Value range can be different from 255. Other common ranges are 1 (sigmoid) and 2 (tanh).
    if val_range is None:
        if torch.max(img1) > 128:
            max_val = 255
        else:
            max_val = 1

        if torch.min(img1) < -0.5:
            min_val = -1
        else:
            min_val = 0
        L = max_val - min_val
    else:
        L = val_range

    padd = 0
    (_, channel, height, width) = img1.size()
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity

    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    if size_average:
        ret = ssim_map.mean()
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)

    if full:
        return ret, cs
    return ret

class MINSSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True, val_range=None):
        super(MINSSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.val_range = val_range

        # Assume 1 channel for SSIM
        self.channel = 1
        self.window = create_window(window_size)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.dtype == img1.dtype:
            window = self.window
        else:
            window = create_window(self.window_size, channel).to(img1.device).type(img1.dtype)
            self.window = window
            self.channel = channel

        return 1 - ssim(img1, img2, window=window, window_size=self.window_size, size_average=self.size_average)

@LOSS_REGISTRY.register()
class SSIM(nn.Module):
    def __init__(self,loss_weight=1.0):
        super(SSIM, self).__init__()
        self.minssim = MINSSIM()
        self.loss_weight = loss_weight

    def forward(self, x, target):
        loss = self.minssim(x, target)
        return self.loss_weight * loss

@LOSS_REGISTRY.register()
class FrequencyLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(FrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, x, target):
        if IS_HIGH_VERSION:
            x_fft = torch.fft.rfft2(x)
            x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)

            target_fft = torch.fft.rfft2(target)
            target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        else:
            x_fft = torch.rfft(x, signal_ndim=2, normalized=False, onesided=True)
            target_fft = torch.rfft(target, signal_ndim=2, normalized=False, onesided=True)
        return self.loss_weight * self.criterion(x_fft, target_fft)

@LOSS_REGISTRY.register()
class pramidFrequencyLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(pramidFrequencyLoss, self).__init__()
        self.loss_weight = loss_weight
        self.criterion = torch.nn.L1Loss(reduction=reduction)

    def forward(self, x, target, factor, max_levels):

        x = imgtoDivisible(x,factor)
        target = imgtoDivisible(target, factor)

        pyr_x = gau_pyramid(x, max_levels)
        pyr_target = gau_pyramid(target, max_levels)

        for i in range(max_levels):
            if IS_HIGH_VERSION:
                x_fft = torch.fft.rfft2(pyr_x[i])
                x_fft = torch.stack([x_fft.real, x_fft.imag], dim=-1)

                target_fft = torch.fft.rfft2(pyr_target[i])
                target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
            else:
                x_fft = torch.rfft(pyr_x[i], signal_ndim=2, normalized=False, onesided=True)
                target_fft = torch.rfft(pyr_target[i], signal_ndim=2, normalized=False, onesided=True)
        loss = 0
        for i in range(max_levels):
            loss += self.loss_weight * self.criterion(x_fft[i], target_fft[i])
        return loss


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero.
            Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class Gradient_Loss(nn.Module):
    def __init__(self, loss_weight=1.0, device='cuda'):
        super(Gradient_Loss, self).__init__()
        self.loss_weight = loss_weight
        self.sobel_filter_X = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_Y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).reshape((1, 1, 3, 3))
        self.sobel_filter_X = torch.from_numpy(self.sobel_filter_X).float().to(device)
        self.sobel_filter_Y = torch.from_numpy(self.sobel_filter_Y).float().to(device)

    def forward(self, output, gt):
        b, c, h, w = output.size()

        output_X_c, output_Y_c = [], []
        gt_X_c, gt_Y_c = [], []
        for i in range(c):
            output_grad_X = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            output_grad_Y = F.conv2d(output[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)
            gt_grad_X = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_X, bias=None, stride=1, padding=1)
            gt_grad_Y = F.conv2d(gt[:, i:i + 1, :, :], self.sobel_filter_Y, bias=None, stride=1, padding=1)

            output_X_c.append(output_grad_X)
            output_Y_c.append(output_grad_Y)
            gt_X_c.append(gt_grad_X)
            gt_Y_c.append(gt_grad_Y)

        output_X = torch.cat(output_X_c, dim=1)
        output_Y = torch.cat(output_Y_c, dim=1)
        gt_X = torch.cat(gt_X_c, dim=1)
        gt_Y = torch.cat(gt_Y_c, dim=1)

        grad_loss = torch.mean(torch.abs(output_X - gt_X)) + torch.mean(torch.abs(output_Y - gt_Y))

        return self.loss_weight * grad_loss


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

        Args:
            loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super(WeightedTVLoss, self).forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super(WeightedTVLoss, self).forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss

@LOSS_REGISTRY.register()
class AdversarialLoss(nn.Module):
    def __init__(self, use_cpu=False, num_gpu=1, gan_type='WGAN_GP', gan_k=1,
                 lr_dis=1e-4, train_crop_size=40, loss_weight=1.0):

        super(AdversarialLoss, self).__init__()
        self.loss_weight = loss_weight
        self.gan_type = gan_type
        self.gan_k = gan_k
        self.device = torch.device('cpu' if use_cpu else 'cuda')
        self.discriminator = Discriminator(train_crop_size * 4).to(self.device)
        # if (num_gpu > 1):
        #     self.discriminator = nn.DataParallel(self.discriminator, list(range(num_gpu)))
        if (gan_type in ['WGAN_GP', 'GAN']):
            self.optimizer = optim.Adam(
                self.discriminator.parameters(),
                betas=(0, 0.9), eps=1e-8, lr=lr_dis
            )
        else:
            raise SystemExit('Error: no such type of GAN!')

        self.bce_loss = torch.nn.BCELoss().to(self.device)

        # if (D_path):
        #     self.logger.info('load_D_path: ' + D_path)
        #     D_state_dict = torch.load(D_path)
        #     self.discriminator.load_state_dict(D_state_dict['D'])
        #     self.optimizer.load_state_dict(D_state_dict['D_optim'])

    def forward(self, fake, real):
        fake_detach = fake.detach()

        for _ in range(self.gan_k):
            self.optimizer.zero_grad()
            d_fake = self.discriminator(fake_detach)
            d_real = self.discriminator(real)
            if (self.gan_type.find('WGAN') >= 0):
                loss_d = (d_fake - d_real).mean()
                if self.gan_type.find('GP') >= 0:
                    epsilon = torch.rand(real.size(0), 1, 1, 1).to(self.device)
                    epsilon = epsilon.expand(real.size())
                    hat = fake_detach.mul(1 - epsilon) + real.mul(epsilon)
                    hat.requires_grad = True
                    d_hat = self.discriminator(hat)
                    gradients = torch.autograd.grad(
                        outputs=d_hat.sum(), inputs=hat,
                        retain_graph=True, create_graph=True, only_inputs=True
                    )[0]
                    gradients = gradients.view(gradients.size(0), -1)
                    gradient_norm = gradients.norm(2, dim=1)
                    gradient_penalty = 10 * gradient_norm.sub(1).pow(2).mean()
                    loss_d += gradient_penalty

            elif (self.gan_type == 'GAN'):
                valid_score = torch.ones(real.size(0), 1).to(self.device)
                fake_score = torch.zeros(real.size(0), 1).to(self.device)
                real_loss = self.bce_loss(torch.sigmoid(d_real), valid_score)
                fake_loss = self.bce_loss(torch.sigmoid(d_fake), fake_score)
                loss_d = (real_loss + fake_loss) / 2.

            # Discriminator update
            loss_d.backward()
            self.optimizer.step()

        d_fake_for_g = self.discriminator(fake)
        if (self.gan_type.find('WGAN') >= 0):
            loss_g = -d_fake_for_g.mean()
        elif (self.gan_type == 'GAN'):
            loss_g = self.bce_loss(torch.sigmoid(d_fake_for_g), valid_score)

        # Generator loss
        return loss_g*self.loss_weight

    def state_dict(self):
        D_state_dict = self.discriminator.state_dict()
        D_optim_state_dict = self.optimizer.state_dict()
        return D_state_dict, D_optim_state_dict

@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram


@LOSS_REGISTRY.register()
class GANLoss(nn.Module):
    """Define GAN loss.

    Args:
        gan_type (str): Support 'vanilla', 'lsgan', 'wgan', 'hinge'.
        real_label_val (float): The value for real label. Default: 1.0.
        fake_label_val (float): The value for fake label. Default: 0.0.
        loss_weight (float): Loss weight. Default: 1.0.
            Note that loss_weight is only for generators; and it is always 1.0
            for discriminators.
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(GANLoss, self).__init__()
        self.gan_type = gan_type
        self.loss_weight = loss_weight
        self.real_label_val = real_label_val
        self.fake_label_val = fake_label_val

        if self.gan_type == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif self.gan_type == 'lsgan':
            self.loss = nn.MSELoss()
        elif self.gan_type == 'wgan':
            self.loss = self._wgan_loss
        elif self.gan_type == 'wgan_softplus':
            self.loss = self._wgan_softplus_loss
        elif self.gan_type == 'hinge':
            self.loss = nn.ReLU()
        else:
            raise NotImplementedError(f'GAN type {self.gan_type} is not implemented.')

    def _wgan_loss(self, input, target):
        """wgan loss.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return -input.mean() if target else input.mean()

    def _wgan_softplus_loss(self, input, target):
        """wgan loss with soft plus. softplus is a smooth approximation to the
        ReLU function.

        In StyleGAN2, it is called:
            Logistic loss for discriminator;
            Non-saturating loss for generator.

        Args:
            input (Tensor): Input tensor.
            target (bool): Target label.

        Returns:
            Tensor: wgan loss.
        """
        return F.softplus(-input).mean() if target else F.softplus(input).mean()

    def get_target_label(self, input, target_is_real):
        """Get target label.

        Args:
            input (Tensor): Input tensor.
            target_is_real (bool): Whether the target is real or fake.

        Returns:
            (bool | Tensor): Target tensor. Return bool for wgan, otherwise,
                return Tensor.
        """

        if self.gan_type in ['wgan', 'wgan_softplus']:
            return target_is_real
        target_val = (self.real_label_val if target_is_real else self.fake_label_val)
        return input.new_ones(input.size()) * target_val

    def forward(self, input, target_is_real, is_disc=False):
        """
        Args:
            input (Tensor): The input for the loss module, i.e., the network
                prediction.
            target_is_real (bool): Whether the targe is real or fake.
            is_disc (bool): Whether the loss for discriminators or not.
                Default: False.

        Returns:
            Tensor: GAN loss value.
        """
        target_label = self.get_target_label(input, target_is_real)
        if self.gan_type == 'hinge':
            if is_disc:  # for discriminators in hinge-gan
                input = -input if target_is_real else input
                loss = self.loss(1 + input).mean()
            else:  # for generators in hinge-gan
                loss = -input.mean()
        else:  # other gan types
            loss = self.loss(input, target_label)

        # loss_weight is always 1.0 for discriminators
        return loss if is_disc else loss * self.loss_weight


@LOSS_REGISTRY.register()
class MultiScaleGANLoss(GANLoss):
    """
    MultiScaleGANLoss accepts a list of predictions
    """

    def __init__(self, gan_type, real_label_val=1.0, fake_label_val=0.0, loss_weight=1.0):
        super(MultiScaleGANLoss, self).__init__(gan_type, real_label_val, fake_label_val, loss_weight)

    def forward(self, input, target_is_real, is_disc=False):
        """
        The input is a list of tensors, or a list of (a list of tensors)
        """
        if isinstance(input, list):
            loss = 0
            for pred_i in input:
                if isinstance(pred_i, list):
                    # Only compute GAN loss for the last layer
                    # in case of multiscale feature matching
                    pred_i = pred_i[-1]
                # Safe operation: 0-dim tensor calling self.mean() does nothing
                loss_tensor = super().forward(pred_i, target_is_real, is_disc).mean()
                loss += loss_tensor
            return loss / len(input)
        else:
            return super().forward(input, target_is_real, is_disc)


def r1_penalty(real_pred, real_img):
    """R1 regularization for discriminator. The core idea is to
        penalize the gradient on real data alone: when the
        generator distribution produces the true data distribution
        and the discriminator is equal to 0 on the data manifold, the
        gradient penalty ensures that the discriminator cannot create
        a non-zero gradient orthogonal to the data manifold without
        suffering a loss in the GAN game.

        Ref:
        Eq. 9 in Which training methods for GANs do actually converge.
        """
    grad_real = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)[0]
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()
    return grad_penalty


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(fake_img.shape[2] * fake_img.shape[3])
    grad = autograd.grad(outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True)[0]
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_lengths.detach().mean(), path_mean.detach()


def gradient_penalty_loss(discriminator, real_data, fake_data, weight=None):
    """Calculate gradient penalty for wgan-gp.

    Args:
        discriminator (nn.Module): Network for the discriminator.
        real_data (Tensor): Real input data.
        fake_data (Tensor): Fake input data.
        weight (Tensor): Weight tensor. Default: None.

    Returns:
        Tensor: A tensor for gradient penalty.
    """

    batch_size = real_data.size(0)
    alpha = real_data.new_tensor(torch.rand(batch_size, 1, 1, 1))

    # interpolate between real_data and fake_data
    interpolates = alpha * real_data + (1. - alpha) * fake_data
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = discriminator(interpolates)
    gradients = autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    if weight is not None:
        gradients = gradients * weight

    gradients_penalty = ((gradients.norm(2, dim=1) - 1)**2).mean()
    if weight is not None:
        gradients_penalty /= torch.mean(weight)

    return gradients_penalty


@LOSS_REGISTRY.register()
class GANFeatLoss(nn.Module):
    """Define feature matching loss for gans

    Args:
        criterion (str): Support 'l1', 'l2', 'charbonnier'.
        loss_weight (float): Loss weight. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, criterion='l1', loss_weight=1.0, reduction='mean'):
        super(GANFeatLoss, self).__init__()
        if criterion == 'l1':
            self.loss_op = L1Loss(loss_weight, reduction)
        elif criterion == 'l2':
            self.loss_op = MSELoss(loss_weight, reduction)
        elif criterion == 'charbonnier':
            self.loss_op = CharbonnierLoss(loss_weight, reduction)
        else:
            raise ValueError(f'Unsupported loss mode: {criterion}. Supported ones are: l1|l2|charbonnier')

        self.loss_weight = loss_weight

    def forward(self, pred_fake, pred_real):
        num_d = len(pred_fake)
        loss = 0
        for i in range(num_d):  # for each discriminator
            # last output is the final prediction, exclude it
            num_intermediate_outputs = len(pred_fake[i]) - 1
            for j in range(num_intermediate_outputs):  # for each layer output
                unweighted_loss = self.loss_op(pred_fake[i][j], pred_real[i][j].detach())
                loss += unweighted_loss / num_d
        return loss * self.loss_weight


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=True)


class Discriminator(nn.Module):
    def __init__(self, in_size=256):
        super(Discriminator, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.LReLU1 = nn.LeakyReLU(0.2)
        self.conv2 = conv3x3(32, 32, 2)
        self.LReLU2 = nn.LeakyReLU(0.2)
        self.conv3 = conv3x3(32, 64)
        self.LReLU3 = nn.LeakyReLU(0.2)
        self.conv4 = conv3x3(64, 64, 2)
        self.LReLU4 = nn.LeakyReLU(0.2)
        self.conv5 = conv3x3(64, 128)
        self.LReLU5 = nn.LeakyReLU(0.2)
        self.conv6 = conv3x3(128, 128, 2)
        self.LReLU6 = nn.LeakyReLU(0.2)
        self.conv7 = conv3x3(128, 256)
        self.LReLU7 = nn.LeakyReLU(0.2)
        self.conv8 = conv3x3(256, 256, 2)
        self.LReLU8 = nn.LeakyReLU(0.2)
        self.conv9 = conv3x3(256, 512)
        self.LReLU9 = nn.LeakyReLU(0.2)
        self.conv10 = conv3x3(512, 512, 2)
        self.LReLU10 = nn.LeakyReLU(0.2)

        self.fc1 = nn.Linear(32768, 1024)
        self.fc2 = nn.Linear(8192, 1024)
        self.fc3 = nn.Linear(2048, 1024)
        self.LReLU11 = nn.LeakyReLU(0.2)
        self.fc4 = nn.Linear(1024, 1)

    def forward(self, x):
        x = self.LReLU1(self.conv1(x))
        x = self.LReLU2(self.conv2(x))
        x = self.LReLU3(self.conv3(x))
        x = self.LReLU4(self.conv4(x))
        x = self.LReLU5(self.conv5(x))
        x = self.LReLU6(self.conv6(x))
        x = self.LReLU7(self.conv7(x))
        x = self.LReLU8(self.conv8(x))
        x = self.LReLU9(self.conv9(x))
        x = self.LReLU10(self.conv10(x))

        x = x.view(x.size(0), -1)
        if x.size(1) == 32768:
            x = self.LReLU11(self.fc1(x))
        elif x.size(1) == 8192:
            x = self.LReLU11(self.fc2(x))
        else:
            x = self.LReLU11(self.fc3(x))
        x = self.fc4(x)

        return x

class Binary_2channel_Classification(nn.Module):
    def __init__(self):
        super(Binary_2channel_Classification, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=2, kernel_size=3, stride=1, padding=1)
        )

        self.activate = nn.Sigmoid()


    def forward(self, x):
        x = self.body(x)
        out = F.interpolate(x, scale_factor = 8, mode='bilinear')
        out = self.activate(out).cpu().numpy()

        out = torch.from_numpy(np.array(out>0.5, dtype=np.int))
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        return out_2
