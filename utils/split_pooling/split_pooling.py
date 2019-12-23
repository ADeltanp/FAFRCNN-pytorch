import torch as t
import torch.nn as nn
from utils.creator import create_grid
from utils.split_pooling.roi_module import RoIPooling2D


class SplitPooling(nn.Module):
    def __init__(self, base_size=16., scales=t.Tensor([16, 10, 6]), ratios=t.Tensor([0.5, 1, 2]), roi_w=7, roi_h=7):
        """
        :param base_size: (int) length scaling factor from feature to image
        :param scales: (int tuple) grid scales
        :param ratios: (int tuple) grid width to height scale
        """
        super().__init__()
        self.roi = RoIPooling2D(roi_h, roi_w, 1. / 16.)
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios

    def forward(self, img_size, features):
        """
        :param img_size: (int tuple) w * h of images in this batch
        :param features: (t.Tensor) feature extracted by CNN, like vgg16 conv5_3
        :return: (list) large to small grid cells of this batch, shape of num_of_scales * (N, C, roi_w, roi_h)
        """
        grids = create_grid(img_size, self.base_size, self.scales, self.ratios)
        sp_out = list([len(grids) * list()])
        for i in range(features.size()[0]):  # for each batch
            for j, grid in enumerate(grids):
                roi_idx = t.cat((i * t.ones(grid.size(0))[:, None], grid.float()), dim=1).cuda().contiguous()
                sp = self.roi(features, roi_idx)  # (N, Channel, roi_h, roi_w)
                sp_out[j].append(sp)

        return list([t.cat(grid) for grid in sp_out])


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
#     l = sp((img_size, img_size), rand_feature, seed_xy=(0, 0))
#     print(l)
