import torch as t


def smfr(source_model_feat, target_model_feat, anchor_iou):
    mask = t.where(anchor_iou > 0.5, t.ones(source_model_feat.size()), t.zeros(source_model_feat.size()))
    loss = 1. / t.sum(mask) * (t.norm((source_model_feat - target_model_feat) * mask, p=2)) ** 2
    return loss
