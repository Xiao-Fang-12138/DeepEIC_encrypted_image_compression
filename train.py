import os
import argparse
import random
import shutil
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasetRGB import ImageFolder
from utils.dataset_vari import variable_collate, VariableSizeDataset
from utils.my_loss import MyLoss
import logging
import numpy as np
import PIL.Image as Image
from torchvision.transforms import ToPILImage
from typing import Tuple, Union
from torch.utils.tensorboard import SummaryWriter
from network.Layers import ADMMCSNetLayer
from utils.metric import setup_logger, generate_measurement_matrix
from tqdm import tqdm
from network.Encrypt import *
import pdb
from compressai.losses import RateDistortionLoss
from network.Reconstruct import PGEDNN
from torchsummary import summary

def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.cpu().clamp_(0, 1).squeeze())

class SubrateDistortionLoss(nn.Module):

    def __init__(self, lmbda=0.0035, return_type='all'):
        super().__init__()
        self.metric = nn.MSELoss()
        self.lmbda = lmbda
        self.return_type = return_type

    def forward(self, output, target):
        (N, _, H, W) = target.size()
        out = {}
        num_pixels = N * H * W
        out['bpp_loss'] = output['subrate']
        out['mse_loss'] = self.metric(output['x_hat'], target)
        distortion = 255 ** 2 * out['mse_loss']
        out['loss'] = distortion
        if self.return_type == 'all':
            return out
        else:
            return out[self.return_type]

def compute_metrics(a: Union[np.array, Image.Image], b: Union[np.array, Image.Image], max_val: float=255.0) -> Tuple[float, float]:
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)
    a = torch.from_numpy(a.copy()).float().unsqueeze(0)
    if a.dim() == 3:
        a = a.unsqueeze(0)
    elif a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0)
    if b.dim() == 3:
        b = b.unsqueeze(0)
    elif b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)
    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    return p

class AverageMeter:

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class CustomDataParallel(nn.DataParallel):

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)

def configure_optimizers(net, args):
    parameters = {n for (n, p) in net.named_parameters() if not n.endswith('.quantiles') and p.requires_grad}
    params_dict = dict(net.named_parameters())
    optimizer = optim.Adam((params_dict[n] for n in sorted(parameters)), lr=args.learning_rate)
    return optimizer

def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, logger_train, tb_logger, args):
    model.train()
    device = next(model.parameters()).device
    for (i, d) in enumerate(train_dataloader):
        optimizer.zero_grad()
        d = d.to(device)
        (b, c, h, w) = d.shape
        encryptor = ImageEncryptorDecryptor(b, c, h, w)
        output = model(d, encryptor)
        x_hat = output['x_hat']
        out_criterion = criterion(output, d)
        out_criterion['loss'].backward()
        optimizer.step()
        if i % 10 == 0:
            logger_train.info(f"Train epoch {epoch}: [{i * len(d)}/{len(train_dataloader.dataset)} ({100.0 * i / len(train_dataloader):.0f}%)]\tLoss: {out_criterion['loss'].item():.3f} |\tMSE loss: {out_criterion['mse_loss'].item():.3f} |\tBpp loss: {out_criterion['bpp_loss']:.2f} |")
    tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion['loss'].item(), epoch)

def test_epoch(args, epoch, test_dataloader, model, logger_val, criterion, tb_logger):
    model.eval()
    device = next(model.parameters()).device
    i = 0
    psnr = AverageMeter()
    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    with torch.no_grad():
        for d in tqdm(test_dataloader):
            d = d.to(device)
            (b, c, h, w) = d.shape
            encryptor = ImageEncryptorDecryptor(b, c, h, w)
            final_encrypted = encryptor.encrypt(d)
            final_decrypted = encryptor.decrypt(final_encrypted)
            output = model(d, encryptor)
            x_hat = output['x_hat']
            out_criterion = criterion(output, d)
            bpp_loss.update(out_criterion['bpp_loss'])
            loss.update(out_criterion['loss'])
            mse_loss.update(out_criterion['mse_loss'])
            save_dir = os.path.join('experiments', args.experiment, 'images')
            ori_dir = os.path.join(save_dir, 'ori')
            if not os.path.exists(ori_dir):
                os.makedirs(ori_dir)
            oriimg = torch2img(d)
            oriimg.save(os.path.join(ori_dir, '%03d.png' % i))
            final_enc_dir = os.path.join(save_dir, 'final_enc')
            if not os.path.exists(final_enc_dir):
                os.makedirs(final_enc_dir)
            rec_dir = os.path.join(save_dir, 'rec')
            if not os.path.exists(rec_dir):
                os.makedirs(rec_dir)
            inter_enc_dir = os.path.join(save_dir, 'inter_enc')
            if not os.path.exists(inter_enc_dir):
                os.makedirs(inter_enc_dir)
            final_dec_dir = os.path.join(save_dir, 'final_dec')
            if not os.path.exists(final_dec_dir):
                os.makedirs(final_dec_dir)
            inter_dec_dir = os.path.join(save_dir, 'inter_dec')
            if not os.path.exists(inter_dec_dir):
                os.makedirs(inter_dec_dir)
            oriimg = torch2img(d)
            oriimg.save(os.path.join(ori_dir, '%03d.png' % i))
            final_encrypted_img = torch2img(final_encrypted)
            final_encrypted_img.save(os.path.join(final_enc_dir, '%03d.png' % i))
            rec_img = torch2img(x_hat)
            rec_img.save(os.path.join(rec_dir, '%03d.png' % i))
            p = compute_metrics(oriimg, rec_img)
            psnr.update(p)
            i = i + 1
    logger_val.info(f'Test epoch {epoch}: Average losses:\tLoss: {loss.avg:.3f} |\tMSE loss: {mse_loss.avg:.3f} |\tBpp loss: {bpp_loss.avg:.2f} |\tPSNR: {psnr.avg:.6f} |')
    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch)
    return loss.avg

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        dest_filename = filename.replace(filename.split('/')[-1], '_checkpoint_best_loss.pth.tar')
        shutil.copyfile(filename, dest_filename)

def parse_args(argv):
    parser = argparse.ArgumentParser(description='Example training script.')
    parser.add_argument('-d', '--dataset', type=str, required=True, help='Training dataset')
    parser.add_argument('-d_test', '--test_dataset', type=str, required=True, help='Testing dataset')
    parser.add_argument('-e', '--epochs', default=100000, type=int, help='Number of epochs (default: %(default)s)')
    parser.add_argument('-lr', '--learning-rate', default=0.0001, type=float, help='Learning rate (default: %(default)s)')
    parser.add_argument('-n', '--num-workers', type=int, default=4, help='Dataloaders threads (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (default: %(default)s)')
    parser.add_argument('--test-batch-size', type=int, default=1, help='Test batch size (default: %(default)s)')
    parser.add_argument('--patch-size', type=int, nargs=2, default=(256, 256), help='Size of the training patches to be cropped (default: %(default)s)')
    parser.add_argument('--test-patch-size', type=int, nargs=2, default=(1024, 1024), help='Size of the testing patches to be cropped (default: %(default)s)')
    parser.add_argument('--cuda', action='store_true', help='Use cuda')
    parser.add_argument('--save', action='store_true', default=True, help='Save model to disk')
    (parser.add_argument('--checkpoint', type=str, help='Path to a checkpoint'),)
    (parser.add_argument('-exp', '--experiment', type=str, required=True, help='Experiment name'),)
    parser.add_argument('--save-images', action='store_true', default=False, help='Save images to disk')
    (parser.add_argument('--test', action='store_true', default=False, help='test'),)
    (parser.add_argument('--val-freq', default=1, type=int),)
    (parser.add_argument('--n-step', default=12, type=int),)
    (parser.add_argument('--num-blocks', default=3, type=int),)
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    if not os.path.exists(os.path.join('experiments', args.experiment)):
        os.makedirs(os.path.join('experiments', args.experiment))
    setup_logger('train', os.path.join('experiments', args.experiment), 'train_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    setup_logger('val', os.path.join('experiments', args.experiment), 'val_' + args.experiment, level=logging.INFO, screen=True, tofile=True)
    logger_train = logging.getLogger('train')
    logger_val = logging.getLogger('val')
    tb_logger = SummaryWriter(log_dir='./tb_logger/' + args.experiment)
    if not os.path.exists(os.path.join('experiments', args.experiment, 'checkpoints')):
        os.makedirs(os.path.join('experiments', args.experiment, 'checkpoints'))
    train_transforms = transforms.Compose([transforms.RandomCrop(args.patch_size), transforms.ToTensor()])
    test_transforms = transforms.Compose([transforms.CenterCrop(args.test_patch_size), transforms.ToTensor()])
    train_dataset = ImageFolder(args.dataset, split='', transform=train_transforms)
    test_dataset = ImageFolder(args.test_dataset, split='', transform=test_transforms)
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, pin_memory=device == 'cuda')
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, shuffle=False, pin_memory=device == 'cuda')
    net = PGEDNN(in_channels=1, mid_channels=64, n_step=args.n_step, num_blocks=args.num_blocks, subrate=0.0625)
    net = net.to(device)
    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)
    logger_train.info(args)
    optimizer = configure_optimizers(net, args)
    criterion = SubrateDistortionLoss()
    last_epoch = 0
    loss = float('inf')
    best_loss = float('inf')
    if args.checkpoint:
        print('Loading', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        last_epoch = checkpoint['epoch'] + 1
        net.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        optimizer.param_groups[0]['lr'] = args.learning_rate
        best_loss = checkpoint['best_loss']
    if not args.test:
        for epoch in range(last_epoch, args.epochs):
            logger_train.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
            train_one_epoch(net, criterion, train_dataloader, optimizer, epoch, logger_train, tb_logger, args)
            if epoch % args.val_freq == 0:
                loss = test_epoch(args, epoch, test_dataloader, net, logger_val, criterion, tb_logger)
            is_best = loss < best_loss
            best_loss = min(loss, best_loss)
            if args.save:
                save_checkpoint({'epoch': epoch, 'state_dict': net.state_dict(), 'best_loss': best_loss, 'optimizer': optimizer.state_dict()}, is_best, os.path.join('experiments', args.experiment, 'checkpoints', 'net_checkpoint.pth.tar'))
                if is_best:
                    logger_val.info('best checkpoint saved.')
    else:
        loss = test_epoch(args, 0, test_dataloader, net, logger_val, criterion, tb_logger)
if __name__ == '__main__':
    main(sys.argv[1:])