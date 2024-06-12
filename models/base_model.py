import torch
import os
from math import pi

from kornia.color import rgb_to_lab, lab_to_rgb
from numpy import loadtxt, savetxt, unravel_index

from .losses import VGGLoss, PerceptualLossVgg16, GANLoss
from tools.args import BaseOptions
from tools.helper_functions import my_rgb_to_lab, TrainTools, mask_generator

opt = BaseOptions().parse(save=False)
tools = TrainTools(opt)


class BaseModel(torch.nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        self.opt = opt
        # init common losses
        self.gan_loss = GANLoss()
        self.l1_loss = torch.nn.L1Loss()
        self.summed_l1 = torch.nn.L1Loss(reduction='sum')
        self.bce_loss = torch.nn.BCEWithLogitsLoss()
        self.xe_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.summed_mse = torch.nn.MSELoss(reduction='sum')
        if self.opt.use_vgg_loss:
            self.vgg_loss = VGGLoss(detach=False)  # detach only during train?
        if self.opt.use_conv22_loss:
            self.conv22_loss = PerceptualLossVgg16()

    def save_model(self, model, optim, scheduler, name, epoch):
        save_filename = '%s_net_%s.pth' % (epoch, name)
        save_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, save_filename)
        save_dict = {'model_state_dict': model.state_dict()}
        if optim is not None:
            save_dict['optimizer_state_dict'] = optim.state_dict()
        if self.opt.use_scheduler and scheduler is not None:
            save_dict['scheduler_state_dict'] = scheduler.state_dict()
        torch.save(save_dict, save_path)

    def load_model(self, model, optim, scheduler, name, epoch, mode='train'):
        load_filename = '%s_net_%s.pth' % (epoch, name)
        load_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, load_filename)
        checkpoint = torch.load(load_path)
        # model.load_state_dict(checkpoint['model_state_dict'])
        ckpt = checkpoint['model_state_dict']
        model.load_state_dict(ckpt)

        if mode == 'train':
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.opt.use_scheduler:
                try:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                except:
                    print('Failed to load scheduler state dict!')

    # re-normalise to have Lab data in (0,1) range
    def img_to_lab_tensor(self, data):
        data = rgb_to_lab(data)
        data[:, 0, ...] = data[:, 0, ...] / 100.0
        data[:, 1:, ...] = (data[:, 1:, ...] + 128.0) / 255.0
        # if 'norm' in self.opt.name or self.opt.norm_input:
        #     data = data * 2.0 - 1.0
        if self.opt.norm_input:
            data = self.to_tensor(data)
        return data

    # covert Lab tensor to (0,1) img range
    def lab_tensor_to_img(self, data):
        if self.opt.norm_input:
            data = self.to_img(data)
        data[:, 0, ...] = data[:, 0, ...] * 100.0
        data[:, 0, ...].clamp_(0.0, 100.0)
        data[:, 1:, ...] = data[:, 1:, ...] * 255.0 - 128.0
        data[:, 1:, ...].clamp_(-128.0, 127.0)
        data = lab_to_rgb(data)
        return data

    def to_tensor(self, x):  # = convert (0,1) to (-1,1) or imagenet
        if self.opt.norm_input == 'imagenet':
            mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std_dev = [0.5] * 3, [0.5] * 3
        data_len = len(x.shape)
        if data_len == 3:
            c, h, w = x.size(0), x.size(1), x.size(2)
            x_ch0 = (torch.unsqueeze(x[0], 1) - mean[0]) / std_dev[0]
            x_ch1 = (torch.unsqueeze(x[1], 1) - mean[1]) / std_dev[1]
            x_ch2 = (torch.unsqueeze(x[2], 1) - mean[2]) / std_dev[2]
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.view(c, h, w)
        else:
            c, h, w = x.size(1), x.size(2), x.size(3)
            x_ch0 = (torch.unsqueeze(x[:, 0], 1) - mean[0]) / std_dev[0]
            x_ch1 = (torch.unsqueeze(x[:, 1], 1) - mean[1]) / std_dev[1]
            x_ch2 = (torch.unsqueeze(x[:, 2], 1) - mean[2]) / std_dev[2]
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.view(x.size(0), c, h, w)
        return x

    def to_img(self, x):  # = convert from half/imagenet norm to (0,1)
        if self.opt.norm_input == 'imagenet':
            mean, std_dev = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        else:
            mean, std_dev = [0.5] * 3, [0.5] * 3
        data_len = len(x.shape)
        if data_len == 3:  # single image
            c, h, w = x.size(0), x.size(1), x.size(2)
            x_ch0 = torch.unsqueeze(x[0], 0) * std_dev[0] + mean[0]
            x_ch1 = torch.unsqueeze(x[1], 0) * std_dev[1] + mean[1]
            x_ch2 = torch.unsqueeze(x[2], 0) * std_dev[2] + mean[2]
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.clamp(0, 1)
            x = x.view(c, h, w)
        else:  # batch of data
            c, h, w = x.size(1), x.size(2), x.size(3)
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * std_dev[0] + mean[0]
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * std_dev[1] + mean[1]
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * std_dev[2] + mean[2]
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
            x = x.clamp(0, 1)
            x = x.view(x.size(0), c, h, w)
        return x

    def uniform_data(self, data):
        data = data - data.min()  # shift min to 0
        data = data / data.max()  # move max to 1
        return data

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            try:
                m.weight.data.normal_(0.0, 0.02)
            except:
                pass

    def os_loss(self, outA, outB):
        G_os_loss = self.l1_loss(outA['sf'], outB['sf'])
        return G_os_loss

    def sfr_loss(self, out_sf, mask, data_in):
        one_mask = torch.ones_like(mask)
        sf_regionA = (one_mask - mask)
        cut_inputA = sf_regionA * data_in
        cut_sf_outA = sf_regionA * out_sf
        sfr_loss = self.summed_mse(cut_sf_outA, cut_inputA) / (3 * sf_regionA.sum())
        return sfr_loss

    def id_loss(self, data_sf):
        self_out = self.netG.forward(data_sf)
        identity_loss = self.l1_loss(self_out['sf'], data_sf[:, :3, ...])  # constrain data_sf in case cond input
        return identity_loss
