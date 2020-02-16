import torch as t
import math
import random


def pair(source, target, source_num=32, target_num=32):
    """
    :param source: (t.Tensor) source features or grids to be paired. (R, 7, 7) or (R, 4096)
    :param target: (t.Tensor) target features or grids to be paired. (R', 7, 7) or (R', 4096)
    :param source_num: (int) S-S pair num.
    :param target_num: (int) S-T pair num.
    :return: (t.Tensors) both of the shapes are (source_num, 2*), * is shape of source and target
    """
    if len(source) == 0 or len(target) == 0:
        return t.tensor(()), t.tensor(())
    t_capacity = len(target)  # R'
    s_capacity = len(source)  # R

    s_t_pair = list()
    t_shuffle = t.randperm(t_capacity)
    s_shuffle = t.randperm(s_capacity)
    for i in range(t_capacity):
        if len(s_t_pair) >= target_num:
            break
        t_idx = t_shuffle[i]
        for j in range(s_capacity):
            s_idx = s_shuffle[j]
            tensor = (t.cat((source[s_idx], target[t_idx]), dim=-1) if random.random() > 0.5
                      else t.cat((target[t_idx], source[s_idx]), dim=-1))
            s_t_pair.append(tensor)
            if len(s_t_pair) >= target_num:
                break
    s_t_pair = t.stack(s_t_pair)  # (target_num, 14, 7) or (target_num, 4096 * 2)

    s_s_pair = list()
    s_shuffle_2 = t.randperm(s_capacity)
    for i in range(s_capacity):
        if len(s_s_pair) >= source_num:
            break
        s_idx_1 = s_shuffle[i]
        for j in range(s_capacity):
            s_idx_2 = s_shuffle_2[j]
            tensor = t.cat((source[s_idx_1], source[s_idx_2]), dim=-1)
            s_s_pair.append(tensor)
            if len(s_s_pair) >= source_num:
                break
    s_s_pair = t.stack(s_s_pair)  # (source_num, 14, 7) or (source_num, 4096 * 2)

    return s_s_pair.squeeze(), s_t_pair.squeeze()
