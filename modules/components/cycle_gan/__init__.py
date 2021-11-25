import torch
from torch.nn import L1Loss

from .cycle_gan_generator import Generator
from .cycle_gan_discriminator import Discriminator, GANLoss

import argparse


def build_components(args: argparse.Namespace):
    net_G_A = Generator(args.input_nc, args.output_nc, args.ngf, args.ng_blocks)
    net_G_B = Generator(args.input_nc, args.output_nc, args.ngf, args.ng_blocks)
    net_D_A = Discriminator(args.input_nc, args.ndf, args.nd_layers)
    net_D_B = Discriminator(args.input_nc, args.ndf, args.nd_layers)

    cycle_criterion = L1Loss()
    idt_criterion = L1Loss()
    gan_criterion = GANLoss(torch.device("cuda:{}".format(args.gpu_device) if args.cuda else torch.device("cpu")))
    l1_criterion = L1Loss()

    return net_G_A, net_G_B, net_D_A, net_D_B, l1_criterion, cycle_criterion, idt_criterion, gan_criterion
