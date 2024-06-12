import os
import numpy as np
import wandb as wb
import torch

from models.models import SSWShadowModel
from tools.args import BaseOptions
from tools.visualizer import Visualizer
from tools.dataloader import get_loader
from tools.helper_functions import TrainTools, gradient_penalty, mask_generator

opt = BaseOptions().parse()
visualizer = Visualizer(opt)
tools = TrainTools(opt)
train_data = get_loader(opt, 'train')
dataset_size = len(train_data)

ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('# training pairs = %d' % dataset_size)
model = SSWShadowModel().to(my_device)
start_epoch, epoch_iter = tools.manage_ckpts(model)
model.train()

total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq  # how often to show results
save_delta = total_steps % opt.save_latest_freq  # how often to save the model

# adjust display/save freq is not divisible by batch size
if opt.batch_size != 1:
    while opt.display_freq % opt.batch_size:
        opt.display_freq+=1
        opt.save_latest_freq+=1

# if you want w&b logging (previously tensorboard)
if opt.tb_log:
    wb_path = os.path.join(opt.tb_dir, opt.name, 'id.txt')
    if os.path.exists(wb_path):
        wb_id = str(np.loadtxt(wb_path, dtype=str))
    else:
        os.mkdir(os.path.join(opt.tb_dir, opt.name))
        wb_id = wb.util.generate_id()
        np.savetxt(wb_path, [wb_id], fmt='%s')
    wb.init(project=opt.dataset, name=opt.name, id=wb_id, dir=opt.tb_dir+opt.name, resume=True)

for epoch in range(start_epoch, opt.num_epochs + 1):

    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    loss_dict = {}

    for data in train_data:
        data = tools.process_inputs(data, model.img_to_lab_tensor)

        # do critic steps
        if total_steps % (opt.critic_iter + 1) != opt.critic_iter:
            for p in model.netD.parameters():  # reset requires_grad
                p.requires_grad = True  # they are set to False below in netG update
            model.optimizerD.zero_grad()
            D_loss_real, D_loss_fake, gp = model.discriminator_step(data)
            loss_D = D_loss_fake - D_loss_real + gp
            loss_D.backward()
            model.optimizerD.step()
            loss_dict['D_fake'], loss_dict['D_real'] = D_loss_fake, D_loss_real
            loss_G = 0  # in case we accidentally log
        # do G step
        else:
            for p in model.netD.parameters():
                p.requires_grad = False  # to avoid computation
            model.optimizerG.zero_grad()
            out_A, out_B, G_loss_dict = model.generator_step(data)
            loss_G = sum(G_loss_dict.values())   # this will include R losses, if netR used
            loss_dict = {**loss_dict, **G_loss_dict}
            loss_G.backward()
            model.optimizerG.step()

        # ----- log & save -----
        total_steps += opt.batch_size
        epoch_iter += opt.batch_size

        if total_steps % opt.display_freq == display_delta:
            # print out errors
            errors = {k: v.data.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
            errors['D_all'], errors['G_all'] = loss_D, loss_G  # add combined to see general trend
            visualizer.print_current_errors(epoch, epoch_iter, errors)

            # save images
            if opt.lab_input:  # undo Lab
                for k in ['inA', 'inB', 'sfA', 'sfB']:
                    data[k] = model.lab_tensor_to_img(data[k])
                out_A['sf'], out_B['sf'] = model.lab_tensor_to_img(out_A['sf']), model.lab_tensor_to_img(out_B['sf'])
            visualizer.quick_save(data['inA'][0], epoch, 'inA')
            visualizer.quick_save(data['inB'][0], epoch, 'inB')
            visualizer.quick_save(data['sfA'][0], epoch, 'in_sfA')
            visualizer.quick_save(data['sfB'][0], epoch, 'in_sfB')
            visualizer.quick_save(out_A['sf'][0], epoch, 'out_sfA')
            visualizer.quick_save(out_B['sf'][0], epoch, 'out_sfB')

            if opt.tb_log:
                imgs_to_log = [data['inA'], data['inB'], out_A['sf'], out_B['sf']]
                wb.log({'Images': wb.Image(visualizer.log_images(imgs_to_log), caption="L-R: input / estim-SF ")},
                       step=total_steps, commit=False)
                wb.log(loss_dict, step=total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save_models('latest')
            np.savetxt(ckpt_path, (epoch, epoch_iter), delimiter=',', fmt='%d')

        if epoch_iter >= dataset_size:
            break

    # save model at the end of the epoch
    if epoch % opt.save_epoch_freq == 0:  # and opt.save_models:
        print('saving the model at the end of epoch %d' % epoch)
        model.save_models('latest')
        model.save_models(epoch)
        np.savetxt(ckpt_path, (epoch + 1, 0), delimiter=',', fmt='%d')

    if opt.use_scheduler:
        model.schedulerG.step()
        model.schedulerD.step()

