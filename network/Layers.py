import numpy as np
import torch.nn as nn
import torchpwl
from scipy.io import loadmat
from os.path import join
import os
import torch

class ADMMCSNetLayer(nn.Module):

    def __init__(self, mask, in_channels: int=1, out_channels: int=128, kernel_size: int=5):
        super(ADMMCSNetLayer, self).__init__()
        self.rho = nn.Parameter(torch.tensor([0.1]), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([1.0]), requires_grad=True)
        self.mask = mask
        self.re_org_layer = ReconstructionOriginalLayer(self.rho, self.mask)
        self.conv1_layer = ConvolutionLayer1(in_channels, out_channels, kernel_size)
        self.nonlinear_layer = NonlinearLayer()
        self.conv2_layer = ConvolutionLayer2(out_channels, in_channels, kernel_size)
        self.min_layer = MinusLayer()
        self.multiple_org_layer = MultipleOriginalLayer(self.gamma)
        self.re_update_layer = ReconstructionUpdateLayer(self.rho, self.mask)
        self.add_layer = AdditionalLayer()
        self.multiple_update_layer = MultipleUpdateLayer(self.gamma)
        self.re_final_layer = ReconstructionFinalLayer(self.rho, self.mask)
        layers = []
        layers.append(self.re_org_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_org_layer)
        for i in range(8):
            layers.append(self.re_update_layer)
            layers.append(self.add_layer)
            layers.append(self.conv1_layer)
            layers.append(self.nonlinear_layer)
            layers.append(self.conv2_layer)
            layers.append(self.min_layer)
            layers.append(self.multiple_update_layer)
        layers.append(self.re_update_layer)
        layers.append(self.add_layer)
        layers.append(self.conv1_layer)
        layers.append(self.nonlinear_layer)
        layers.append(self.conv2_layer)
        layers.append(self.min_layer)
        layers.append(self.multiple_update_layer)
        layers.append(self.re_final_layer)
        self.cs_net = nn.Sequential(*layers)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1_layer.conv.weight = torch.nn.init.normal_(self.conv1_layer.conv.weight, mean=0, std=1)
        self.conv2_layer.conv.weight = torch.nn.init.normal_(self.conv2_layer.conv.weight, mean=0, std=1)
        self.conv1_layer.conv.weight.data = self.conv1_layer.conv.weight.data * 0.025
        self.conv2_layer.conv.weight.data = self.conv2_layer.conv.weight.data * 0.025

    def forward(self, y):
        x = self.cs_net(y)
        x = y + (1 - self.mask.cuda()) * x
        return x

class ReconstructionOriginalLayer(nn.Module):

    def __init__(self, rho, mask):
        super(ReconstructionOriginalLayer, self).__init__()
        self.rho = rho
        self.ATBBTA = mask

    def forward(self, x):
        ATBBTA = self.ATBBTA
        denom = torch.add(ATBBTA.cuda(), self.rho)
        a = 1e-06
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(x, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        cs_data = dict()
        cs_data['input'] = x
        cs_data['conv1_input'] = orig_output3
        return cs_data

class ReconstructionUpdateLayer(nn.Module):

    def __init__(self, rho, mask):
        super(ReconstructionUpdateLayer, self).__init__()
        self.rho = rho
        self.ATBBTA = mask

    def forward(self, x):
        minus_output = x['minus_output']
        multiple_output = x['multi_output']
        input = x['input']
        ATBBTA = self.ATBBTA
        number = torch.add(input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output)))
        denom = torch.add(ATBBTA.cuda(), self.rho)
        a = 1e-06
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x['re_mid_output'] = orig_output3
        return x

class ReconstructionFinalLayer(nn.Module):

    def __init__(self, rho, mask):
        super(ReconstructionFinalLayer, self).__init__()
        self.rho = rho
        self.ATBBTA = mask

    def forward(self, x):
        minus_output = x['minus_output']
        multiple_output = x['multi_output']
        input = x['input']
        ATBBTA = self.ATBBTA
        number = torch.add(input, self.rho * torch.fft.fft2(torch.sub(minus_output, multiple_output)))
        denom = torch.add(ATBBTA.cuda(), self.rho)
        a = 1e-06
        value = torch.full(denom.size(), a).cuda()
        denom = torch.where(denom == 0, value, denom)
        orig_output1 = torch.div(1, denom)
        orig_output2 = torch.mul(number, orig_output1)
        orig_output3 = torch.fft.ifft2(orig_output2)
        x['re_final_output'] = orig_output3
        return x['re_final_output']

class ConvolutionLayer1(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer1, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2), stride=1, dilation=1, bias=True)

    def forward(self, x):
        conv1_input = x['conv1_input']
        x['conv1_output'] = self.conv(conv1_input)
        return x

class ConvolutionLayer2(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int):
        super(ConvolutionLayer2, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=int((kernel_size - 1) / 2), stride=1, dilation=1, bias=True)

    def forward(self, x):
        nonlinear_output = x['nonlinear_output']
        x['conv2_output'] = self.conv(nonlinear_output)
        return x

class NonlinearLayer(nn.Module):

    def __init__(self):
        super(NonlinearLayer, self).__init__()
        self.pwl = torchpwl.PWL(num_channels=128, num_breakpoints=101)

    def forward(self, x):
        conv1_output = x['conv1_output']
        x['nonlinear_output'] = self.pwl(conv1_output)
        return x

class MinusLayer(nn.Module):

    def __init__(self):
        super(MinusLayer, self).__init__()

    def forward(self, x):
        minus_input = x['conv1_input']
        conv2_output = x['conv2_output']
        output = torch.sub(minus_input, conv2_output)
        x['minus_output'] = output
        return x

class AdditionalLayer(nn.Module):

    def __init__(self):
        super(AdditionalLayer, self).__init__()

    def forward(self, x):
        mid_output = x['re_mid_output']
        multi_output = x['multi_output']
        output = torch.add(mid_output, multi_output)
        x['conv1_input'] = output
        return x

class MultipleOriginalLayer(nn.Module):

    def __init__(self, gamma):
        super(MultipleOriginalLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        org_output = x['conv1_input']
        minus_output = x['minus_output']
        output = torch.mul(self.gamma, torch.sub(org_output, minus_output))
        x['multi_output'] = output
        return x

class MultipleUpdateLayer(nn.Module):

    def __init__(self, gamma):
        super(MultipleUpdateLayer, self).__init__()
        self.gamma = gamma

    def forward(self, x):
        multiple_output = x['multi_output']
        re_mid_output = x['re_mid_output']
        minus_output = x['minus_output']
        output = torch.add(multiple_output, torch.mul(self.gamma, torch.sub(re_mid_output, minus_output)))
        x['multi_output'] = output
        return x