import torch as t
import math
import random


def pair(source, target, source_num=128, target_num=128):
    t_capacity = len(target)
    s_capacity = len(source)
    t_repeat = float(target_num) / float(t_capacity)
    s_repeat = float(source_num) / float(s_capacity)

    assert t_repeat > s_capacity, "Insufficient amount for S-T pairing."
    assert source_num > sum(i for i in range(s_capacity + 1)), "Insufficient amount for S-S pairing."

    t_ceil = int(math.ceil(t_repeat))
    s_t_pair = list()
    s_rand_idx = t.randperm(s_capacity)

    for i in range(t_capacity):
        for j in range(t_ceil):
            idx = s_rand_idx[(i * t_ceil + j) % s_capacity]
            tensor = (t.cat((source[idx], target[i])) if random.random() > 0.5
                      else t.cat((target[i], source[idx])))
            s_t_pair.append(tensor)

    s_t_pair = t.stack(s_t_pair)
    s_t_rand_idx = t.randperm(s_t_pair.size(0))
    idx = s_t_rand_idx[:target_num]
    s_t_pair = s_t_pair[idx]

    s_s_pair = list()
    s_rand_idx = t.randperm(s_capacity)

    for i in range(int(math.ceil(s_repeat))):
        for j in range(s_capacity):
            if len(s_s_pair) > source_num:
                break
            tensor = t.cat((source[s_rand_idx[j]], source[s_rand_idx[j + i]]))
            s_s_pair.append(tensor)

    s_s_pair = t.stack(s_s_pair)
    s_s_rand_idx = t.randperm(s_s_pair.size(0))
    idx = s_s_rand_idx[:source_num]
    s_s_pair = s_s_pair[idx]

    return s_s_pair, s_t_pair
