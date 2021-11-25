import torch
import random


class ImagePool:
    def __init__(self, size: int = 50, img_size: tuple[int, int] = (256, 256)):
        super(ImagePool, self).__init__()
        self.size = size
        self.cursor = 0
        self.is_full = False
        self.queue = torch.zeros((size, 3, *img_size))
        self.queue.requires_grad = False

    def put(self, imgs: torch.Tensor):
        B = imgs.shape[0]
        imgs.requires_grad = False
        if B + self.cursor > self.size:
            end_ind = self.size - self.cursor
            self.queue[self.cursor:self.cursor + end_ind] = imgs[:end_ind]
            self.queue[:B - end_ind] = imgs[end_ind:]
            self.cursor = B - end_ind
            self.is_full = True
        else:
            self.queue[self.cursor:self.cursor + B] = imgs
            self.cursor += B

    def get(self, batch_size):
        if self.is_full:
            ind = torch.randint(low=0, high=50, size=(batch_size, ))
            return self.queue[ind]
        else:
            ind = torch.randint(low=0, high=self.cursor, size=(batch_size, ))
            return self.queue[ind]


