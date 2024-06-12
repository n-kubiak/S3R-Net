import torch
from torch.optim.lr_scheduler import StepLR

from tools.helper_functions import mask_generator, my_rgb_to_lab, gradient_penalty
from .networks import InOutGenerator, MultiscaleDiscriminator
from .base_model import BaseModel


# MAIN MODEL
class SSWShadowModel(BaseModel):
    def __init__(self):
        super(SSWShadowModel, self).__init__()

        ### GENERATOR
        self.netG = InOutGenerator(input_nc=3, output_nc=3)
        self.netG.apply(self.weights_init)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(0.0, 0.9))
        self.schedulerG = StepLR(self.optimizerG, step_size=self.opt.scheduler_step, gamma=self.opt.scheduler_gamma) \
            if self.opt.use_scheduler else None

        ### DISCRIMINATOR
        self.netD = MultiscaleDiscriminator(input_nc=3, affine=True, conv_bias=True)
        self.netD.apply(self.weights_init)
        self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=self.opt.lr, betas=(0.0, 0.9))
        self.schedulerD = StepLR(self.optimizerD, step_size=self.opt.scheduler_step, gamma=self.opt.scheduler_gamma) \
            if self.opt.use_scheduler else None

    def runG(self, x):
        return self.netG.forward(x)

    def runD(self, x):
        return self.netD.forward(x)

    def discriminator_step(self, data):
        # real
        D_real_A = self.runD(data['sfA'])
        D_real_B = self.runD(data['sfB'])
        D_loss_real = D_real_A[0][0].mean() + D_real_B[0][0].mean()
        if len(D_real_A) == 2:
            D_loss_real += (D_real_A[1][0].mean() + D_real_B[1][0].mean())

        # fake
        gen_out_A = self.runG(data['inA'].detach())
        gen_out_B = self.runG(data['inB'].detach())
        D_fake_A = self.runD(gen_out_A['sf'])
        D_fake_B = self.runD(gen_out_B['sf'])
        D_loss_fake = D_fake_A[0][0].mean() + D_fake_B[0][0].mean()
        if len(D_fake_A) == 2:
            D_loss_fake += (D_fake_A[1][0].mean() + D_fake_B[1][0].mean())

        # gp
        gp = gradient_penalty(self, data['sfA'], gen_out_A['sf'], device=data['sfA'].device)
        gp += gradient_penalty(self, data['sfB'], gen_out_B['sf'], device=data['sfA'].device)

        return D_loss_real, D_loss_fake, gp

    def generator_step(self, data):
        loss_dict = {}
        out_A, out_B = self.runG(data['inA']), self.runG(data['inB'])
        maskA, maskB = mask_generator(data['inA'], out_A['sf']), mask_generator(data['inB'], out_B['sf'])

        # calculate generator losses
        if self.opt.use_os_loss:
            loss_dict['G_os'] = self.os_loss(out_A, out_B) * self.opt.use_os_loss

        if self.opt.use_sfr_loss:
            loss_dict['G_sfr'] = (self.sfr_loss(out_A['sf'], maskA, data['inA']) + self.sfr_loss(out_B['sf'],
                maskB, data['inB'])) * self.opt.use_sfr_loss

        if self.opt.use_identity_loss:  # identity loss (SF image will remain SF)
            loss_dict['G_id'] = self.id_loss(data['sfA']) * self.opt.use_identity_loss

        if self.opt.use_vgg_loss:
            vgg_loss = self.vgg_loss(out_A['sf'], out_B['sf'])
            loss_dict['G_vgg'] = vgg_loss * self.opt.use_vgg_loss

        if self.opt.use_conv22_loss:
            conv22_loss = self.conv22_loss(out_A['sf'], data['inA']) + self.conv22_loss(out_B['sf'], data['inB'])
            loss_dict['G_conv22'] = conv22_loss * self.opt.use_conv22_loss

        # calculate GAN losses
        G_fake_A, G_fake_B = self.runD(out_A['sf']), self.runD(out_B['sf'])
        G_loss_fake = -torch.mean(G_fake_A[0][0]) - torch.mean(G_fake_B[0][0])
        if len(G_fake_A) == 2:
            G_loss_fake -= (torch.mean(G_fake_A[1][0]) + torch.mean(G_fake_B[1][0]))
        loss_dict['G_fake'] = -abs(G_loss_fake) if self.opt.abs_fake else G_loss_fake

        return out_A, out_B, loss_dict

    def save_models(self, epoch):
        self.save_model(self.netG, self.optimizerG, self.schedulerG, 'G', epoch)
        self.save_model(self.netD, self.optimizerD, self.schedulerD, 'D', epoch)


class ShadowModel_Inference(BaseModel):
    def __init__(self):
        super(ShadowModel_Inference, self).__init__()
        self.netG = InOutGenerator(input_nc=3, output_nc=3, ngf=64, n_downsampling=3, n_blocks=4)
        self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=self.opt.lr, betas=(0.0, 0.999))
        self.schedulerG = None

    def forward(self, data):
        out_data = self.netG.forward(data['in'])
        ref = data['sf'].detach().clone()
        mask = data['sm'].detach()
        fake_image = out_data['sf'].detach().clone()

        # if Lab, convert back to RGB & then convert to special Lab params from the original matlab eval script
        if self.opt.lab_input:
            ref = self.lab_tensor_to_img(ref)
            fake_image = self.lab_tensor_to_img(fake_image)
        lab_ref = my_rgb_to_lab(ref)
        lab_fake = my_rgb_to_lab(fake_image)

        # ORIGINAL EVAL
        abs_L = abs(lab_ref[0][0] - lab_fake[0][0])
        abs_a = abs(lab_ref[0][1] - lab_fake[0][1])
        abs_b = abs(lab_ref[0][2] - lab_fake[0][2])

        # overall MAE
        mae_all_L = abs_L.mean()
        mae_all_a = abs_a.mean()
        mae_all_b = abs_b.mean()

        # if testing on a dataset with SMs available
        if 'sm' in data:
            mask = (mask > 0.5).float()  # binarize in case there are any outliers
            mask = mask[0][0]
            sf_mask = 1-mask

            # shadow-free pixels and MAE
            sf_L = abs_L * sf_mask
            sf_a = abs_a * sf_mask
            sf_b = abs_b * sf_mask
            mae_sf_L = sf_L.sum() / sf_mask.sum()
            mae_sf_a = sf_a.sum() / sf_mask.sum()
            mae_sf_b = sf_b.sum() / sf_mask.sum()

            # shadowed pixels and MAE
            s_L = abs_L * mask
            s_a = abs_a * mask
            s_b = abs_b * mask
            mae_s_L = s_L.sum() / mask.sum()
            mae_s_a = s_a.sum() / mask.sum()
            mae_s_b = s_b.sum() / mask.sum()

        # accumulate MAE all/S/NS per channel, like in org script
        loss_dict = {'RMSE_A': [mae_all_L, mae_all_a, mae_all_b]}
        if 'sm' in data:
            loss_dict['RMSE_S'] = [mae_s_L, mae_s_a, mae_s_b]
            loss_dict['RMSE_N'] = [mae_sf_L, mae_sf_a, mae_sf_b]
        else:  # if testing deshadowing performance on a dataset without SMs, return 0s
            loss_dict['RMSE_S'] = [0, 0, 0]
            loss_dict['RMSE_N'] = [0, 0, 0]

        return out_data, loss_dict

