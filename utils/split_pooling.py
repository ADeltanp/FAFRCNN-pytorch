import torch as t
import torch.nn as nn
from torchvision.ops import roi_pool
from utils.creator import create_grid


class SplitPooling(nn.Module):
    def __init__(self, base_size=16., scales=t.Tensor([16, 10, 6]), ratios=t.Tensor([0.5, 1, 2]), roi_h=7, roi_w=7):
        """
        :param base_size: (int) length scaling factor from feature to image
        :param scales: (int tuple) grid scales
        :param ratios: (int tuple) grid width to height scale
        """
        super().__init__()
        self.roi_out_size = (roi_h, roi_w)
        # self.roi = RoIPooling2D(roi_h, roi_w, 1. / 16.)
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios

    def forward(self, img_size, features):
        """
        :param img_size: (int tuple) w * h of images in this batch
        :param features: (t.Tensor) feature extracted by CNN, like vgg16 conv5_3
        :return: (list of t.Tensor) large to small grid cells of this batch,
        shape of num_of_scales * (N * G', C, roi_w, roi_h)
        """
        # different scale and ratio grid, grids[i][j] for scale i ratio j
        # shape of grids[i][j] is (G, 4), totally G' grids of one scale (i)
        grids = create_grid(img_size, self.base_size, self.scales, self.ratios)
        sp_out = [[] for i in range(len(grids))]
        n = features.size(0)
        for i, grid in enumerate(grids):
            g = grid.size(0)
            n_grid = t.cat([grid.float()] * n)
            idx = t.cat([t.ones((g, 1)) * k for k in range(n)])
            roi_idx = t.cat([idx, n_grid], dim=1).cuda().contiguous()
            sp = roi_pool(features, roi_idx, self.roi_out_size, 1. / 16.)  # (N * G', Channel, roi_h, roi_w)
            sp_out[i].append(sp)
        return [t.cat(sp) for sp in sp_out]


# with t.no_grad():
#     sp = SplitPooling(scales=t.tensor([4]), ratios=t.tensor([1]))
#     rand_feature = t.tensor([[1, 1, 1, 1, 2, 2, 2, 2],
#                             [1, 1, 1, 1, 2, 2, 2, 2],
#                             [1, 1, 1, 1, 2, 2, 2, 2],
#                             [1, 1, 1, 1, 2, 2, 2, 2],
#                             [3, 3, 3, 3, 4, 4, 4, 4],
#                             [3, 3, 3, 3, 4, 4, 4, 4],
#                             [3, 3, 3, 3, 4, 4, 4, 4],
#                             [3, 3, 3, 3, 4, 4, 4, 4]])[None, None, :].float().cuda()
#     img_size = 8 * 16
#     l = sp((img_size, img_size), rand_feature)  # TODO NEED SEED X Y
#     print(l)
