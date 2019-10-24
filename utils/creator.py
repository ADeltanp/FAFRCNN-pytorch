import torch as t
import random


def create_grid(img_size, base_size=16., scales=[16, 10, 6], ratios=[0.5, 1, 2]):
    """
    :param img_size: (int tuple) w * h of the image
    :param base_size: (int) length in image corresponding to one unit feature, 16 for vgg16 conv5_3
    :param scales: (int tuple) grid cell size
    :param ratios: (int tuple) grid width height ratio

    :return: (t.Tensor) grids of different sizes and ratios, for the same size and ratio, its shape is (G, 4), G is the
    number of grids generated of current size(scale). Each is (x_l_top, y_l_top, x_r_bottom, y_r_b) on the image.
    """
    grids = list()
    for i in range(len(scales)):
        scale_i_grid = list()
        for j in range(len(ratios)):
            w = base_size * scales[i] * (ratios[j] ** 0.5)
            h = base_size * scales[i] / (ratios[j] ** 0.5)
            w = round(w)
            h = round(h)

            sx = random.randint(0, w, (1, ))
            sy = random.randint(0, h, (1, ))
            base_coord = t.Tensor([sx, sy, sx + w, sy + h]).int()

            w_slices = (img_size[0] - sx) // w
            h_slices = (img_size[1] - sy) // h

            x_coord = t.Tensor([sx + w * k for k in range(w_slices)]).int()
            y_coord = t.Tensor([sy + h * k for k in range(h_slices)]).int()
            y_coord, x_coord = t.meshgrid(x_coord, y_coord)

            coord = t.stack((x_coord.flatten(), y_coord.flatten,
                             x_coord.flatten(), y_coord.flatten()), dim=1)
            scale_i_grid.append(base_coord + coord)  # (1, 4) + (C, 4) = (C, 4)

        grids.append(scale_i_grid)  # grids[i] for scale i grid, grids[i][j] for scale i ratio j grid.

    return t.cat(grids[0]), t.cat(grids[1]), t.cat(grids[2])
