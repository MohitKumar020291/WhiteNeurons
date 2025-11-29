import torch
import torch.nn as nn


class Test(nn.Module):
    def __init__(self):
        super().__init__()

        image_size=32 
        patch_size=8 
        in_channels=3
        embed_dim=784

        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size, bias=False)

    def forward(self, x):
        x = self.projection(x)
        print(x.shape)
    

from PIL import Image
import os
import numpy as np

cwd = os.getcwd()
image_path = os.path.join(cwd, "puppy.jpeg")
assert os.path.exists(image_path)
image = Image.open(image_path)
image_numpy = np.array(image)
image_torch_tensor = torch.from_numpy(image_numpy)


output = Test()