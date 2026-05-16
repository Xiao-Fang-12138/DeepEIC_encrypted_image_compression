import torch
import torch.nn as nn
import torch.nn.functional as F
from network.Encrypt import ImageEncryptorDecryptor
import numpy as np
from network.EntropyModel import EntropyBottleneck, GaussianConditional
import pdb
from torch.autograd import Variable
from PIL import Image

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2)

def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, output_padding=stride - 1, padding=kernel_size // 2)

class ResidualBlock(nn.Module):

    def __init__(self, channels, has_BN=False):
        super(ResidualBlock, self).__init__()
        self.has_BN = has_BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        if has_BN:
            self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        if self.has_BN:
            residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        if self.has_BN:
            residual = self.bn2(residual)
        return x + residual

class BasicUnit(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, num_blocks=8):
        super(BasicUnit, self).__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, padding=3), nn.PReLU())
        self.residual_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.residual_blocks.append(ResidualBlock(64, has_BN=False))
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.PReLU())
        self.block3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, input):
        block1 = self.block1(input)
        for block in self.residual_blocks:
            block1 = block(block1)
        block2 = self.block2(block1)
        block3 = self.block3(block1 + block2)
        return block3

class SamplingNet(nn.Module):

    def __init__(self, blocksize, subrate, num_blocks=0, height=8, width=8):
        super(SamplingNet, self).__init__()
        self.sampling = nn.Conv2d(3 * blocksize ** 2, int(np.round(3 * blocksize * blocksize * subrate)), 1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(3, 3, 3, stride=1, padding=1, bias=False)
        self.height = height
        self.width = width
        self.output_channel = int(np.round(blocksize * blocksize * subrate))
        self.pixelunshuffle = nn.PixelUnshuffle(downscale_factor=blocksize)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pixelunshuffle(x)
        output = self.sampling(x)
        return output

class UpsamplingNet(nn.Module):

    def __init__(self, blocksize, subrate, num_blocks=0):
        super(UpsamplingNet, self).__init__()
        self.upsampling = nn.Conv2d(int(np.round(3 * blocksize * blocksize * subrate)), 3 * blocksize * blocksize, 1, stride=1, padding=0)
        self.pixelshuffle = nn.PixelShuffle(upscale_factor=blocksize)

    def forward(self, x):
        x = self.upsampling(x)
        x = self.pixelshuffle(x)
        return x

class PGEDBlock(nn.Module):

    def __init__(self, in_channels, mid_channels, num_blocks, blocksize, subrate):
        super(PGEDBlock, self).__init__()
        self.prox = BasicUnit(in_channels, mid_channels, in_channels, num_blocks)
        self.blocksize = blocksize
        self.sampling = SamplingNet(blocksize, subrate)
        self.upsampling = UpsamplingNet(blocksize, subrate)

    def forward(self, X, Y, encryptor):
        Z = encryptor.encrypt(X)
        Y_hat = self.sampling(Z)
        Y_residual = Y - Y_hat
        Y_residual = Y - Y_hat
        Z_grad = self.upsampling(Y_residual)
        X_grad = encryptor.decrypt(Z_grad)
        X = self.prox(X + X_grad)
        return X

class BatchQuantization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        B = x.shape[0]
        x_view = x.view(B, -1)
        min_vals = x_view.min(dim=1).values
        max_vals = x_view.max(dim=1).values
        scales = torch.where(max_vals == min_vals, torch.ones_like(max_vals), (max_vals - min_vals) / 255.0)
        min_vals = min_vals.view(B, 1, 1, 1)
        scales = scales.view(B, 1, 1, 1)
        x_scaled = (x - min_vals) / scales
        x_quantized = (torch.round(x_scaled) - x_scaled).detach() + x_scaled
        x_quantized = torch.clamp(x_quantized, 0, 255).to(torch.uint8)
        return (x_quantized, scales.squeeze(), min_vals.squeeze())

class BatchDequantization(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x_quantized, scales, min_vals):
        B = x_quantized.shape[0]
        scales = scales.view(B, 1, 1, 1)
        min_vals = min_vals.view(B, 1, 1, 1)
        x_dequantized = x_quantized.float() * scales + min_vals
        return x_dequantized.to(torch.float32)

class PGEDNN(nn.Module):

    def __init__(self, in_channels, mid_channels, n_step, num_blocks, subrate=0.1, blocksize=16, bitdepth=8):
        super(PGEDNN, self).__init__()
        M = 512
        N = 384
        self.entropy_bottleneck = EntropyBottleneck(channels=N)
        self.gaussian_conditional = GaussianConditional(None)
        self.PGED_blocks = nn.ModuleList([PGEDBlock(in_channels, mid_channels, num_blocks, blocksize, subrate) for i in range(n_step)])
        self.sampling = SamplingNet(blocksize, subrate)
        self.upsampling = UpsamplingNet(blocksize, subrate)
        self.blocksize = blocksize
        self.subrate = subrate
        self.quantize = BatchQuantization()
        self.dequantize = BatchDequantization()

    def forward(self, X, encryptor):
        device = 'cuda'
        (b, c, h, w) = X.shape
        final_encrypted = encryptor.encrypt(X)
        subrate = self.subrate
        y = self.sampling(final_encrypted)
        (y_quantized, scale, min_val) = self.quantize(y)
        y_hat = self.dequantize(y_quantized, scale, min_val)
        z_hat = self.upsampling(y_hat)
        x_hat = encryptor.decrypt(z_hat)
        for i in range(len(self.PGED_blocks)):
            x_hat = self.PGED_blocks[i](x_hat, y_hat, encryptor)
        return {'x_hat': x_hat, 'subrate': subrate}

    def sampling_test(self, X, encryptor):
        device = 'cuda'
        (b, c, h, w) = X.shape
        final_encrypted = encryptor.encrypt(X)
        y = self.sampling(final_encrypted)
        return y
if __name__ == '__main__':
    input = torch.rand(4, 1, 256, 256)
    model = SamplingNet(32, 0.5)
    output = model(input)
    print(output.shape)