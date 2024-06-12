from math import sqrt

import numpy as np
import os
from skimage.filters import threshold_otsu

import torch
from torchvision import transforms


class TrainTools:
    def __init__(self, opt):
        self.opt = opt
        self.ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
        self.my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def manage_ckpts(self, model):
        try:
            model.load_model(model.netD, model.optimizerD, model.schedulerD, 'D', self.opt.which)
            print('got D')
            model.load_model(model.netG, model.optimizerG, model.schedulerG, 'G', self.opt.which)
            print('got G')

            if self.opt.which == 'latest':
                start_epoch, epoch_iter = np.loadtxt(self.ckpt_path, delimiter=',', dtype=int)
            else:
                start_epoch, epoch_iter = int(self.opt.which) + 1, 0
            print('Resuming from epoch %d at iteration %d' % (start_epoch, epoch_iter))
        except:
            start_epoch, epoch_iter = 1, 0
            print('No checkpoints found; starting from scratch.')
        return start_epoch, epoch_iter

    # put data on the right device and convert tensor2lab (t2l) if specified
    def process_inputs(self, imgs, t2l=None):
        t2l = t2l or (lambda x: x)  # process data only if tensor2lab (t2l) function given
        if type(imgs) is dict:
            for k, v in imgs.items():
                if 'flip' not in k:
                    imgs[k] = imgs[k].to(self.my_device)
                    if self.opt.lab_input and imgs[k].size(1) > 1 and imgs[k].dim() == 4:
                        imgs[k] = t2l(imgs[k])  # apply Lab conversion only to images (+ not SM)
        else:
            imgs = imgs.to(self.my_device)
            if self.opt.lab_input:
                imgs = t2l(imgs)
        return imgs

# generate masks based on Otsu tresholding
def mask_generator(shadow, shadow_free, grad=False):
    to_pil = transforms.ToPILImage()
    to_gray = transforms.Grayscale(num_output_channels=1)
    im_f = to_gray(to_pil(((shadow_free.data.squeeze(0) + 1.0) * 0.5)))
    im_s = to_gray(to_pil(((shadow.data.squeeze(0) + 1.0) * 0.5)))

    diff = (np.asarray(im_f, dtype='float32') - np.asarray(im_s, dtype='float32'))
    L = threshold_otsu(diff)
    mask = torch.tensor(np.float32(diff >= L)).unsqueeze(0).unsqueeze(0).cuda()  # 0:non-shadow, 1.0:shadow
    mask.requires_grad = grad
    return mask

# WGAN gradient penalty (gp)
def gradient_penalty(model, real_data, fake_data, device='cuda'):
    eps = torch.rand(real_data.shape[0], 1, 1, 1).expand(real_data.size()).to(device)
    interpolated = eps * real_data + ((1 - eps) * fake_data)
    D_interpolated = model.runD(interpolated)

    gradients = torch.autograd.grad(inputs=interpolated, outputs=D_interpolated[0][0],
                                    grad_outputs=torch.ones_like(D_interpolated[0][0]),
                                    create_graph=True, retain_graph=True)[0]

    gradients = gradients.view(gradients.shape[0], -1)  # flatten to average per image (and not per batch)
    gp = ((gradients.norm(2, dim=1) - 1) ** 2)
    gp = gp.sum() / gp.numel() * 10  # for debugging

    if len(D_interpolated) == 2:
        gradients2 = torch.autograd.grad(inputs=interpolated, outputs=D_interpolated[1][0],
                                         grad_outputs=torch.ones_like(D_interpolated[1][0]),
                                         create_graph=True, retain_graph=True)[0]

        gradients2 = gradients2.view(gradients2.shape[0], -1)  # flatten to average per image (and not per batch)
        gradients2 = ((gradients2.norm(2, dim=1) - 1) ** 2)
        gp += gradients2.sum() / gradients2.numel() * 10

    return gp

# RGB->Lab conversion consistent with the original matlab eval script (different than default/kornia!)
def my_rgb_to_lab(image: torch.Tensor) -> torch.Tensor:
    # use kornia's srgb->rgb, update matrix in rgb->xyz and the whitepoint in xyz->lab
    if not isinstance(image, torch.Tensor):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(type(image)))

    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}".format(image.shape))

    # convert from sRGB to RGB
    from kornia.color import rgb_to_linear_rgb
    lin_rgb = rgb_to_linear_rgb(image)

    # convert from RGB to XYZ - values for conversion matrix taken from matlab via:
    # cform = makecform('srgb2xyz'); cform.cdata.cforms{1}.cdata.MatTRC
    # matlab_xyz_matrix = torch.tensor([[0.436065673828125, 0.3851470947265625, 0.14306640625],
    #                                 [0.2224884033203125, 0.7168731689453125, 0.06060791015625],
    #                                 [0.013916015625, 0.097076416015625, 0.7140960693359375]])

    # conversion code adapted from kornia
    r: torch.Tensor = lin_rgb[..., 0, :, :]
    g: torch.Tensor = lin_rgb[..., 1, :, :]
    b: torch.Tensor = lin_rgb[..., 2, :, :]

    x: torch.Tensor = 0.436065673828125 * r + 0.3851470947265625 * g + 0.14306640625 * b
    y: torch.Tensor = 0.2224884033203125 * r + 0.7168731689453125 * g + 0.06060791015625 * b
    z: torch.Tensor = 0.013916015625 * r + 0.097076416015625 * g + 0.7140960693359375 * b

    xyz_im: torch.Tensor = torch.stack([x, y, z], -3)

    # normalize for d50 white point
    # xyz_ref_white = torch.tensor([0.95047, 1.0, 1.08883], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None, None] # whitepoint d65 (default)
    xyz_ref_white = torch.tensor([0.964212, 1.000, 0.825188], device=xyz_im.device, dtype=xyz_im.dtype)[..., :, None,
                    None]  # whitepoint d50
    xyz_normalized = torch.div(xyz_im, xyz_ref_white)

    # proceed with xyz->lab conversion
    threshold = 0.008856
    power = torch.pow(xyz_normalized.clamp(min=threshold), 1 / 3.0)
    scale = 7.787 * xyz_normalized + 4.0 / 29.0
    xyz_int = torch.where(xyz_normalized > threshold, power, scale)

    x: torch.Tensor = xyz_int[..., 0, :, :]
    y: torch.Tensor = xyz_int[..., 1, :, :]
    z: torch.Tensor = xyz_int[..., 2, :, :]

    L: torch.Tensor = (116.0 * y) - 16.0
    a: torch.Tensor = 500.0 * (x - y)
    _b: torch.Tensor = 200.0 * (y - z)

    out: torch.Tensor = torch.stack([L, a, _b], dim=-3)

    return out
