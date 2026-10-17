"""DeepELIC: encrypted sampling and five joint reconstruction stages.

Active tensor names and arithmetic match the released research checkpoints.
The unused entropy models from the experiments have been removed. The model
uses continuous measurements in train mode and 8-bit measurements in eval mode.
"""

import torch
from torch import nn

from .encryption import ImageCipher


class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return x + self.conv2(self.prelu(self.conv1(x)))


class ProximalEstimator(nn.Module):
    def __init__(self, num_blocks):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, 7, padding=3), nn.PReLU())
        self.residual_blocks = nn.ModuleList(ResidualBlock() for _ in range(num_blocks))
        self.block2 = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.PReLU())
        self.block3 = nn.Conv2d(64, 3, 3, padding=1)

    def forward(self, x):
        x = self.block1(x)
        for block in self.residual_blocks:
            x = block(x)
        return self.block3(x + self.block2(x))


class SamplingNet(nn.Module):
    def __init__(self, block_size, measurement_channels):
        super().__init__()
        self.sampling = nn.Conv2d(3 * block_size**2, measurement_channels, 1, bias=False)
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1, bias=False)
        self.pixelunshuffle = nn.PixelUnshuffle(block_size)

    def forward(self, x):
        return self.sampling(self.pixelunshuffle(self.conv1(x)))


class UpsamplingNet(nn.Module):
    def __init__(self, block_size, measurement_channels):
        super().__init__()
        self.upsampling = nn.Conv2d(measurement_channels, 3 * block_size**2, 1)
        self.pixelshuffle = nn.PixelShuffle(block_size)

    def forward(self, x):
        return self.pixelshuffle(self.upsampling(x))


class ReconstructionStage(nn.Module):
    def __init__(self, num_blocks, block_size, measurement_channels):
        super().__init__()
        self.prox = ProximalEstimator(num_blocks)
        self.sampling = SamplingNet(block_size, measurement_channels)
        self.upsampling = UpsamplingNet(block_size, measurement_channels)

    def forward(self, x, y, cipher):
        residual = y - self.sampling(cipher.encrypt(x))
        # Preserve the research implementation's affine inverse, including NPT.
        update = cipher.decrypt(self.upsampling(residual))
        return self.prox(x + update)


def quantize_measurements(y):
    """Per-image min/max 8-bit quantization, with FP32 scale and minimum."""
    flat = y.flatten(1)
    minimum = flat.min(1).values.reshape(-1, 1, 1, 1)
    maximum = flat.max(1).values.reshape(-1, 1, 1, 1)
    scale = torch.where(maximum == minimum, torch.ones_like(maximum), (maximum - minimum) / 255)
    symbols = ((y - minimum) / scale).round().clamp(0, 255).to(torch.uint8)
    return symbols, scale, minimum


def dequantize_measurements(symbols, scale, minimum):
    return symbols.float() * scale + minimum


class DeepELIC(nn.Module):
    def __init__(self, subrate=0.25, block_size=16, stages=5, num_blocks=3):
        super().__init__()
        if not 0 < subrate <= 1 or min(block_size, stages, num_blocks) < 1:
            raise ValueError("Invalid architecture configuration")
        self.config = dict(subrate=subrate, block_size=block_size, stages=stages, num_blocks=num_blocks)
        self.block_size = block_size
        self.subrate = subrate
        self.measurement_channels = round(3 * block_size**2 * subrate)
        if self.measurement_channels < 1:
            raise ValueError("subrate produces no measurement channels")
        self.PGED_blocks = nn.ModuleList(
            ReconstructionStage(num_blocks, block_size, self.measurement_channels)
            for _ in range(stages)
        )
        self.sampling = SamplingNet(block_size, self.measurement_channels)
        self.upsampling = UpsamplingNet(block_size, self.measurement_channels)

    def forward(self, x, cipher=None):
        if x.ndim != 4 or x.shape[1] != 3 or any(s % self.block_size for s in x.shape[-2:]):
            raise ValueError("Expected RGB BCHW input with height/width divisible by block_size")
        if cipher is None:
            cipher = ImageCipher(x.shape, x.device)
        y = self.sampling(cipher.encrypt(x))
        payload = {}
        if self.training:
            y_hat = y
        else:
            symbols, scale, minimum = quantize_measurements(y)
            y_hat = dequantize_measurements(symbols, scale, minimum)
            payload = dict(symbols=symbols, scale=scale, minimum=minimum)
        x_hat = cipher.decrypt(self.upsampling(y_hat))
        for stage in self.PGED_blocks:
            x_hat = stage(x_hat, y_hat, cipher)
        return dict(x_hat=x_hat, subrate=self.subrate, quantized=not self.training, **payload)
