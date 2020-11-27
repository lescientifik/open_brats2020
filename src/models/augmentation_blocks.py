"""The strange way used to perform data augmentation during the Brats 2020 challenge...

Be aware, any batch size above 1 could fail miserably (in an unpredicted way).
"""

from random import randint, random, sample, uniform

import torch
from torch import nn


class DataAugmenter(nn.Module):
    """Performs random flip and rotation batch wise, and reverse it if needed.
    Works"""

    def __init__(self, p=0.5, noise_only=False, channel_shuffling=False, drop_channnel=False):
        super(DataAugmenter, self).__init__()
        self.p = p
        self.transpose = []
        self.flip = []
        self.toggle = False
        self.noise_only = noise_only
        self.channel_shuffling = channel_shuffling
        self.drop_channel = drop_channnel

    def forward(self, x):
        with torch.no_grad():
            if random() < self.p:
                x = x * uniform(0.9, 1.1)
                std_per_channel = torch.stack(
                    list(torch.std(x[:, i][x[:, i] > 0]) for i in range(x.size(1)))
                )
                noise = torch.stack([torch.normal(0, std * 0.1, size=x[0, 0].shape) for std in std_per_channel]
                                    ).to(x.device)
                x = x + noise
                if random() < 0.2 and self.channel_shuffling:
                    new_channel_order = sample(range(x.size(1)), x.size(1))
                    x = x[:, new_channel_order]
                    print("channel shuffling")
                if random() < 0.2 and self.drop_channel:
                    x[:, sample(range(x.size(1)), 1)] = 0
                    print("channel Dropping")
                if self.noise_only:
                    return x
                self.transpose = sample(range(2, x.dim()), 2)
                self.flip = randint(2, x.dim() - 1)
                self.toggle = not self.toggle
                new_x = x.transpose(*self.transpose).flip(self.flip)
                return new_x
            else:
                return x

    def reverse(self, x):
        if self.toggle:
            self.toggle = not self.toggle
            if isinstance(x, list):  # case of deep supervision
                seg, deeps = x
                reversed_seg = seg.flip(self.flip).transpose(*self.transpose)
                reversed_deep = [deep.flip(self.flip).transpose(*self.transpose) for deep in deeps]
                return reversed_seg, reversed_deep
            else:
                return x.flip(self.flip).transpose(*self.transpose)
        else:
            return x
