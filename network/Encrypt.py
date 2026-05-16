import torch
import numpy as np
import pdb

def randperm(n):
    return torch.randperm(n)

def scramble(index, original_image):
    index = index.to('cuda')
    (b, c, h, w) = original_image.shape
    vec_reshape_image = original_image.view(b, c, h * w)
    enc_vec_reshape_image = torch.gather(vec_reshape_image, 2, index.expand(b, c, h * w)).to('cuda')
    encrypted_image = enc_vec_reshape_image.view(b, c, h, w)
    return encrypted_image

def unscramble(index, encrypted_image):
    index = index.to('cuda')
    (b, c, h, w) = encrypted_image.shape
    vec_reshape_encrypted_image = encrypted_image.view(b, c, h * w)
    vec_reshape_image = torch.zeros((b, c, h * w), dtype=torch.float32).to('cuda')
    vec_reshape_image.scatter_(2, index.expand(b, c, h * w), vec_reshape_encrypted_image)
    decrypted_image = vec_reshape_image.view(b, c, h, w)
    return decrypted_image

def randRBI(original_image):
    (b, c, h, w) = original_image.shape
    RBI_matrix = torch.randint(0, 2, (b, c, h, w), dtype=torch.torch.float32, device='cuda')
    return RBI_matrix

def negative_positive_transform(original_image, RBI_matrix):
    encrypted_image_NPT = torch.where(RBI_matrix == 1, 256 - original_image, original_image).type(torch.torch.float32)
    return encrypted_image_NPT

def negative_positive_inverse_transform(encrypted_image, RBI_matrix):
    decrypted_image_NPT = torch.where(RBI_matrix == 1, 256 - encrypted_image, encrypted_image)
    return decrypted_image_NPT

def encrypt(x):
    (b, c, h, w) = x.shape
    index = randperm(h * w)
    intermediate_encrypted = scramble(index, x * 255.0)
    RBI_matrix = randRBI(intermediate_encrypted)
    final_encrypted = negative_positive_transform(intermediate_encrypted, RBI_matrix)
    return (final_encrypted, RBI_matrix, index)

def decrypt(final_encrypted, RBI_matrix, index):
    intermediate_decrypted = negative_positive_inverse_transform(final_encrypted, RBI_matrix)
    final_decrypted = unscramble(index, intermediate_decrypted) / 255.0
    return final_decrypted
import torch

class ImageEncryptorDecryptor:

    def __init__(self, b, c, h, w):
        self.index = torch.randperm(h * w).to('cuda')
        self.mask = torch.randint(0, 256, (b, c, h, w), dtype=torch.float32, device='cuda')
        self.RBI_matrix = torch.randint(0, 2, (b, c, h, w), dtype=torch.float32, device='cuda')

    def scramble(self, index, original_image):
        (b, c, h, w) = original_image.shape
        vec_reshape_image = original_image.view(b, c, h * w)
        index = index.to(original_image.device)
        enc_vec_reshape_image = torch.gather(vec_reshape_image, 2, index.expand(b, c, h * w))
        encrypted_image = enc_vec_reshape_image.view(b, c, h, w)
        return encrypted_image

    def unscramble(self, index, encrypted_image):
        (b, c, h, w) = encrypted_image.shape
        vec_reshape_encrypted_image = encrypted_image.view(b, c, h * w)
        index = index.to(encrypted_image.device)
        vec_reshape_image = torch.zeros((b, c, h * w), dtype=torch.float32, device=encrypted_image.device)
        vec_reshape_image.scatter_(2, index.expand(b, c, h * w), vec_reshape_encrypted_image)
        decrypted_image = vec_reshape_image.view(b, c, h, w)
        return decrypted_image

    def random_mask(self, original_image):
        (b, c, h, w) = original_image.shape
        noisy_img = (original_image * 255.0 + self.mask) % 256
        encrypted_image = noisy_img / 255.0
        return encrypted_image

    def random_demask(self, original_image):
        (b, c, h, w) = original_image.shape
        noisy_img = (original_image * 255.0 - self.mask + 256.0) % 256
        decrypted_image = noisy_img / 255.0
        return decrypted_image

    def negative_positive_transform(self, original_image, RBI_matrix):
        encrypted_image_NPT = torch.where(RBI_matrix == 1.0, 1.0 - original_image, original_image).type(torch.float32)
        return encrypted_image_NPT

    def negative_positive_inverse_transform(self, encrypted_image, RBI_matrix):
        decrypted_image_NPT = torch.where(RBI_matrix == 1.0, 1.0 - encrypted_image, encrypted_image).type(torch.float32)
        return decrypted_image_NPT

    def encrypt(self, x):
        (b, c, h, w) = x.shape
        final_encrypted = self.scramble(self.index, x)
        final_encrypted = self.negative_positive_transform(final_encrypted, self.RBI_matrix)
        return final_encrypted

    def decrypt(self, final_encrypted):
        final_decrypted = self.negative_positive_inverse_transform(final_encrypted, self.RBI_matrix)
        final_decrypted = self.unscramble(self.index, final_decrypted)
        return final_decrypted

def main():
    kk = torch.rand(1, 3, 512, 512).cuda()
    model = ImageEncryptorDecryptor(1, 3, 512, 512)
    output = model.encrypt(kk)
    outputoutput = model.decrypt(output)
    print(kk)
    print(kk - outputoutput)
if __name__ == '__main__':
    main()