import torch
from torch import nn
from torchvision import models


class ClassifierLoss(nn.Module):
    def __init__(self):
        super(ClassifierLoss, self).__init__()
        from .convnext import ConvNeXt
        self.net = ConvNeXt(in_chans=3, num_classes=2, depths=[3, 3, 6], dims=[128, 256, 512], first_reduction=4, head=True)
        net_path = '/vol/research/relighting/code/ShadowRel/checkpoints/classifier_lr0.001_wd0.0005_best/30_net_G.pth'
        self.net.load_state_dict(torch.load(net_path)['model_state_dict'])
        self.net.eval()
        self.xe = nn.CrossEntropyLoss()

    def forward(self, x):
        label_size = x.size(0)
        with torch.no_grad():
            out = self.net(x)
            ones = torch.ones(label_size, dtype=torch.int64, device=out.device)
        # breakpoint()
        loss = self.xe(out, ones)
        print(loss)
        return loss

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True):
        super(GANLoss, self).__init__()
        self.real_label = 1.0
        self.fake_label = 0.0
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = torch.FloatTensor
        # if use_lsgan:  # encourage images to move closer to the decision boundary (LSGAN)
        self.loss = nn.MSELoss()
        # else:  # standard GAN loss, default for pix2pix
        #     self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):  # create a label image that's all 1's or 0's
        if target_is_real:
            create_label = ((self.real_label_var is None) or (self.real_label_var.numel() != input.numel()))
            if create_label:
                self.real_label_var = self.Tensor(input.size()).fill_(self.real_label)
                # self.real_label_var.requires_grad = False --> requires_grad is False by default
        else:
            create_label = ((self.fake_label_var is None) or (self.fake_label_var.numel() != input.numel()))
            if create_label:
                self.fake_label_var = self.Tensor(input.size()).fill_(self.fake_label)

        target_tensor = self.real_label_var if target_is_real else self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                target_tensor = target_tensor.cuda()
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)


class VGGLoss(nn.Module):
    def __init__(self, detach=True):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.detach = detach

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        if self.detach:  # y_vgg detached by default
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        else:  # for os loss we want both grads i guess?
            for i in range(len(x_vgg)):
                loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i])
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        #breakpoint()
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out


class PerceptualLossVgg16(nn.Module):
    def __init__(self, vgg=None, weights=[1.0], indices=[22], normalize=True):
        super(PerceptualLossVgg16, self).__init__()
        self.vgg = Vgg16().cuda() if vgg is None else vgg
        self.criterion = nn.L1Loss()
        self.weights = weights
        self.indices = indices

    def forward(self, out, inp):
        x_vgg, y_vgg = self.vgg(out, self.indices), self.vgg(inp, self.indices)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss


class Vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16, self).__init__()
        self.vgg_pretrained_features = models.vgg16(pretrained=True).features
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X, indices=None):
        if indices is None:
            indices = [3, 8, 15, 22]  # assuming 0 starting index!
        out = []
        for i in range(indices[-1]+1):
            X = self.vgg_pretrained_features[i](X)
            if i in indices:
                out.append(X)
        return out
