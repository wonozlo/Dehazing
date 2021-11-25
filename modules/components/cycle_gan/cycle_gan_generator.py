import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, input_nc: int, output_nc: int, ngf: int, n_blocks: int):
        super(Generator, self).__init__()
        self.conv_in = nn.Conv2d(input_nc, ngf, kernel_size=(7, 7), padding=3, padding_mode='reflect', bias=False)
        self.norm_in = nn.InstanceNorm2d(ngf)

        self.down_block1 = DownBlock(ngf*(2**0))
        self.down_block2 = DownBlock(ngf*(2**1))

        res_blocks = []
        for i in range(n_blocks):
            res_blocks.append(ResBlock(ngf*(2**2)))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.up_block1 = UpBlock(ngf*(2**2))
        self.up_block2 = UpBlock(ngf*(2**1))

        self.conv_out = nn.Conv2d(ngf, output_nc, kernel_size=(7, 7), padding='same', padding_mode='reflect', bias=False)

        self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm_in(self.conv_in(x)))
        down1 = self.down_block1(x)
        down2 = self.down_block2(down1)
        x = self.res_blocks(down2)
        x = self.up_block1(torch.cat([x, down2], dim=1))
        x = self.up_block2(torch.cat([x, down1], dim=1))
        x = self.tanh(self.conv_out(x))
        return x


class ResBlock(nn.Module):
    def __init__(self, nf: int):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, kernel_size=(3, 3), padding='same', padding_mode="reflect", bias=False)
        self.norm1 = nn.InstanceNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, kernel_size=(3, 3), padding='same', padding_mode="reflect", bias=False)
        self.norm2 = nn.InstanceNorm2d(nf)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        return out + x


class DownBlock(nn.Module):
    def __init__(self, nf: int):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(nf, nf * 2, kernel_size=(3, 3), stride=(2, 2), padding=1, bias=False)
        self.norm = nn.InstanceNorm2d(nf * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))


class UpBlock(nn.Module):
    def __init__(self, nf: int):
        super(UpBlock, self).__init__()
        self.conv = nn.ConvTranspose2d(2 * nf, nf//2, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1),
                                       output_padding=(1, 1), bias=False)
        self.norm = nn.InstanceNorm2d(nf//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.norm(self.conv(x)))
