import torch
from torch import nn


class Discriminator(nn.Module):
    def __init__(self, input_nc: int, ndf: int, n_layers: int = 3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(input_nc, ndf, kernel_size=(4, 4), stride=(2, 2), padding=1)
        conv_blocks = []
        for i in range(n_layers - 1):
            conv_blocks.append(ConvBlock(ndf*(2**i)))
        self.conv_blocks = nn.Sequential(*conv_blocks)
        self.conv2 = nn.Conv2d(ndf*(2**(n_layers - 1)), ndf*(2**n_layers), kernel_size=(4, 4), stride=(1, 1), padding=(1, 1),
                               bias=False)
        self.conv3 = nn.Conv2d(ndf*(2**n_layers), 1, kernel_size=(4, 4), stride=(1, 1))
        self.in_norm = nn.InstanceNorm2d(ndf*(2**n_layers))
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu(self.conv1(x))
        x = self.conv_blocks(x)
        x = self.lrelu(self.in_norm(self.conv2(x)))
        x = self.conv3(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, nf: int):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(nf, nf*2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False)
        self.in_norm = nn.InstanceNorm2d(nf*2)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lrelu(self.in_norm(self.conv(x)))


class GANLoss(nn.Module):
    def __init__(self, device):
        super(GANLoss, self).__init__()
        self.device = device
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))
        self.bce = nn.MSELoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction).to(self.device)

    def __call__(self, prediction, target_is_real):
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        loss = self.bce(prediction, target_tensor)
        return loss
