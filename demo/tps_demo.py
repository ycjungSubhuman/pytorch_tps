import torch
import torchvision
from skimage import io
import os, sys

sys.path.append(
    os.path.dirname(
        os.path.dirname(os.path.realpath(__file__))))
import tps


pts_before = [
    [[0,0],
     [255,0],
     [0,255],
     [255,255],
     [0,150],
     [150,0]]
]

pts_after = [
    [[0,0],
     [255,0],
     [0,255],
     [255,255],
     [0,200],
     [200,0]]
]

pts_before_t = torch.Tensor(pts_before)
pts_after_t = torch.Tensor(pts_after)

im = (torch.from_numpy(io.imread('test.png').astype('float32'))
      .permute(2, 0, 1)
      .unsqueeze(0))/256

warp = tps.WarpTPS()
newim = warp(im, pts_before_t[0:1], pts_after_t[0:1])

for i in range(len(pts_after[0])):
    newim[0,:,pts_after[0][i][0], pts_after[0][i][1]] = torch.Tensor([1,0,0])

torchvision.utils.save_image(
    torchvision.utils.make_grid(newim), 'result.png')


