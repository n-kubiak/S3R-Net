import os
from tqdm import tqdm

import torch

from models.models import ShadowModel_Inference
from tools.dataloader import get_loader
from tools.visualizer import Visualizer
from tools.args import BaseOptions
from tools.helper_functions import TrainTools, mask_generator

opt = BaseOptions().parse(save=False)
visualizer = Visualizer(opt)
tt = TrainTools(opt)

eval_data = get_loader(opt, 'test')
dataset_size = len(eval_data)
print('# evaluation images = %d' % dataset_size)
ckpt_path = os.path.join(opt.checkpoints_dir, opt.name)

# define D & G, losses and optim
my_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ShadowModel_Inference().to(my_device)
model.load_model(model.netG, model.optimizerG, model.schedulerG, 'G', opt.which, mode='test')
model.eval()

print('currently running: ', opt.name)
total_loss = {'RMSE_A': [0,0,0], 'RMSE_S': [0,0,0], 'RMSE_N': [0,0,0]}
with torch.no_grad():
    for i, data in tqdm(enumerate(eval_data)):
        data = tt.process_inputs(data, model.img_to_lab_tensor)
        out, loss_dict = model(data)
        for x in range(3):
            total_loss['RMSE_A'][x] += loss_dict['RMSE_A'][x]
            total_loss['RMSE_S'][x] += loss_dict['RMSE_S'][x]
            total_loss['RMSE_N'][x] += loss_dict['RMSE_N'][x]

for k, v in total_loss.items():
    if isinstance(total_loss[k], list):
        total_loss[k] = sum([t/dataset_size for t in v])
    else:
        total_loss[k] = v/dataset_size

# print out errors
errors = {k: round(v.data.item(),4) if isinstance(v, torch.Tensor) else round(v,4) for k, v in total_loss.items()}
visualizer.print_test_errors(errors)
print(errors)

if opt.lab_input:
    data['sf'] = model.lab_tensor_to_img(data['sf'])
    data['in'] = model.lab_tensor_to_img(data['in'])
    out['sf'] = model.lab_tensor_to_img(out['sf'])
# save as PIL Images - better than quick_save/torchvision's random re-norm
visualizer.better_save(data['sf'][0], 'test', f'ref_sf')
visualizer.better_save(data['in'][0], 'test', f'in')
visualizer.better_save(out['sf'][0], 'test', f'out_sf')
