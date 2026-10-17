"""Global random permutation followed by the negative-positive transform."""

import torch


class ImageCipher:
    """One ephemeral GRP/NPT key, shared by every unfolding stage.

    The permutation is shared across RGB channels and batch elements, while
    NPT bits are independent. Inputs and outputs use floating point RGB [0, 1].
    This is the paper's research transform, not a cryptographic key service.
    """

    def __init__(self, shape, device="cpu", generator=None):
        b, c, h, w = shape
        self.shape = tuple(shape)
        self.index = torch.randperm(h * w, generator=generator).to(device)
        self.bits = torch.randint(0, 2, shape, generator=generator).to(device).bool()

    def _permute(self, x, inverse=False):
        if tuple(x.shape) != self.shape:
            raise ValueError("Cipher key and image shapes must match")
        b, c, h, w = x.shape
        flat = x.reshape(b, c, h * w)
        index = self.index.expand(b, c, h * w)
        if inverse:
            return torch.zeros_like(flat).scatter(2, index, flat).reshape_as(x)
        return flat.gather(2, index).reshape_as(x)

    def encrypt(self, x):
        z = self._permute(x)
        return torch.where(self.bits, 1.0 - z, z)

    def decrypt(self, z):
        return self._permute(torch.where(self.bits, 1.0 - z, z), inverse=True)
