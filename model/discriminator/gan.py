import torch.nn as nn


class GANInstance(nn.Module):
    def __init__(self, class_num, head_feat_len=4096):
        super().__init__()
        self.fc1 = nn.Linear(2 * head_feat_len, 10 * class_num)
        self.fc2 = nn.Linear(10 * class_num, 10 * class_num)
        self.fc3 = nn.Linear(10 * class_num, 2 * class_num)

    def forward(self, cat_feat):
        x = self.fc1(cat_feat)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


class GANImage(nn.Module):
    def __init__(self, roi_w, roi_h):
        super().__init__()
        self.fc1 = nn.Linear(2 * roi_w * roi_h, 10)
        self.fc2 = nn.Linear(10, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, cat_grid):
        x = self.fc1(cat_grid)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
