import argparse
import torch
import os


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # WGAN params
        self.parser.add_argument('--abs_fake', action='store_true', help='minus abs G fake?')
        self.parser.add_argument('--critic_iter', type=int, default=5, help='critic iters per each G iter')
        self.parser.add_argument('--gp_lambda', type=int, default=10, help='multiplier for the gradient penalty')

        # hyperparams
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
        self.parser.add_argument('--lr', type=float, default=0.0005, help='learning rate')
        self.parser.add_argument('--use_scheduler', action='store_true', help='use StepLR scheduler for both optimizers')
        self.parser.add_argument('--scheduler_step', type=int, default=10)
        self.parser.add_argument('--scheduler_gamma', type=float, default=0.1)

        # data-related params
        self.parser.add_argument('--lab_input', type=int, default=1, help='train on Lab space data')
        self.parser.add_argument('--norm_input', type=str, default='half', help='apply norm to input data - half/imagenet')
        self.parser.add_argument('--dataset', type=str, default='istd', help='pick your dataset of choice - istd/aistd')
        self.parser.add_argument('--data_root', type=str, default='/vol/research/relighting/datasets/')
        self.parser.add_argument('--img_size', type=int, default=480)
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

        # loss weightings
        self.parser.add_argument('--use_vgg_loss', type=float, default=2.0)
        self.parser.add_argument('--use_sfr_loss', type=float, default=5.0, help='shadow-free region loss, (1-mask)')
        self.parser.add_argument('--use_conv22_loss', type=float, default=2.0)
        self.parser.add_argument('--use_identity_loss', type=float, default=1.0)
        self.parser.add_argument('--use_os_loss', type=float, default=1.0)

        # logging, checkpointing, etc
        self.parser.add_argument('--name', type=str, default='default_name')
        self.parser.add_argument('--num_epochs', type=int, default=30)
        self.parser.add_argument('--from_scratch', action='store_true', help='train from scratch and not from latest')
        self.parser.add_argument('--which', type=str, default='latest', help='which epoch to load (for ckpts)')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints/')
        self.parser.add_argument('--tb_log', type=int, default=0, help='use tensorboard logging -- now w&b')
        self.parser.add_argument('--tb_dir', type=str, default='./logs/')
        self.parser.add_argument('--display_freq', type=int, default=2004, help='frequency of showing training results')
        self.parser.add_argument('--save_latest_freq', type=int, default=5010, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=15, help='frequency of saving checkpoints at the end of epochs')

        self.initialized = True

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.save = save
        args = vars(self.opt)

        # save to the disk
        if self.save:
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            if not os.path.exists(expr_dir):
                os.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'options.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')

        return self.opt
