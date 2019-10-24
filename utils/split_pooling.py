import torch as t
import torch.nn as nn
from .creator import create_grid
from .roi_module import RoIPooling2D


class SplitPooling(nn.Module):
    def __init__(self, base_size=16., scales=t.Tensor([16, 10, 6]), ratios=t.Tensor([0.5, 1, 2])):
        """
        :param base_size: (int) length scaling factor from feature to image
        :param scales: (int tuple) grid scales
        :param ratios: (int tuple) grid width to height scale
        """
        self.roi = RoIPooling2D(7, 7, 1. / 16.)
        self.base_size = base_size
        self.scales = scales
        self.ratios = ratios
        super().__init__()

    def forward(self, img_size, features):
        """
        :param img_size: (int tuple) w * h of images in this batch
        :param features: (t.Tensor) feature extracted by CNN, like vgg16 conv5_3
        :return: (lists of t.Tensor) large to small grid cells of this batch
        """
        large, mid, small = create_grid(img_size, self.base_size, self.scales, self.ratios)
        grids_l = list()
        grids_m = list()
        grids_s = list()
        for i in range(features.size()[0]):
            l_idx = t.cat((i * t.ones(large.size()[0]), large))
            m_idx = t.cat((i * t.ones(mid.size()[0]), mid))
            s_idx = t.cat((i * t.ones(small.size()[0]), small))

            sp_l = self.roi(features, l_idx)
            sp_m = self.roi(features, m_idx)
            sp_s = self.roi(features, s_idx)

            grids_l.append(sp_l)
            grids_m.append(sp_m)
            grids_s.append(sp_s)

        return grids_l, grids_m, grids_s

