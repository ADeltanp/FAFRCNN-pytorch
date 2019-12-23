from __future__ import absolute_import
import os
from collections import namedtuple
import time

import torch as t
from torch import nn
from torch.nn import functional as F
from torchnet.meter import ConfusionMeter, AverageValueMeter

from utils.smfr import smfr
from utils.pair import pair
from utils.split_pooling.split_pooling import SplitPooling

from model.discriminator.gan import GANInstance, GANImage
from model.frcnn.utils import array_tool as at
from model.frcnn.utils.vis_tool import Visualizer
from model.frcnn.utils.config import opt
from model.frcnn.model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator
from model.frcnn.model.transfer_ptl import TransferPTL

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])


class TransferTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn, num_class):
        super(TransferTrainer, self).__init__()
        self.num_class = num_class

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma
        self.split_pooling = SplitPooling()

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.transfer_ptl = TransferPTL()

        self.gan_il = GANImage(7, 7)
        self.gan_im = GANImage(7, 7)
        self.gan_is = GANImage(7, 7)
        self.gan_ins = GANInstance(num_class)
        self.gan_optim = self._init_optim()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        self.gan_optimizer = t.optim.Adam(
            list(self.gan_il.parameters()) +
            list(self.gan_im.parameters()) +
            list(self.gan_is.parameters()) +
            list(self.gan_ins.parameters()))
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    def forward(self, source_imgs, target_imgs, s_bboxes, s_labels, t_bboxes, t_labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            source_imgs (~torch.autograd.Variable): A variable with a batch of source images.
            target_imgs (t.Tensor): A batch of target images.
            s_bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            s_labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            t_labels (~torch.autograd.Variable): A batch of target img labels
                Its shape is (N, R). Background excluded, value ranges from [0, L - 1].
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.
            domain (t.Tensor): A batch of domain labels, (B, 2), (1, 0) is source, (0, 1) is target

        Returns:
            namedtuple of 5 losses
        """
        n = s_bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = source_imgs.shape
        s_img_size = (H, W)
        _, _, h, w = target_imgs.shape
        t_img_size = (h, w)

        s_features = self.faster_rcnn.extractor(source_imgs)
        t_features = self.faster_rcnn.extractor(target_imgs)

        img_gan_loss = self.image_level_gan_loss(s_features, t_features, s_img_size, t_img_size)

        s_rpn_locs, s_rpn_scores, s_rois, s_roi_indices, s_anchor = \
            self.faster_rcnn.rpn(s_features, s_img_size, scale)

        t_rpn_locs, t_rpn_scores, t_rois, t_roi_indices, t_anchor = \
            self.faster_rcnn.rpn(t_features, t_img_size, scale)

        # Since batch size is one, convert variables to singular form
        s_bbox = s_bboxes[0]
        s_label = s_labels[0]
        s_rpn_score = s_rpn_scores[0]
        s_rpn_loc = s_rpn_locs[0]
        s_roi = s_rois

        t_bbox = t_bboxes[0]
        t_label = t_labels[0]
        t_rpn_score = t_rpn_scores[0]  # might be useful
        t_rpn_loc = t_rpn_locs[0]
        t_roi = t_rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of s_rois,
        # consider them as constant input
        s_sample_roi, s_gt_roi_label = self.transfer_ptl(
            s_roi,
            at.tonumpy(s_bbox),
            at.tonumpy(s_label))

        t_sample_roi, t_gt_roi_label = self.transfer_ptl(
            t_roi,
            at.tonumpy(t_bbox),
            at.tonumpy(t_label))

        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(s_sample_roi))
        s_fc7 = self.faster_rcnn.half_forward(
            s_features,
            s_sample_roi,
            sample_roi_index)

        t_fc7 = self.faster_rcnn.half_forward(
            t_features,
            t_sample_roi,
            sample_roi_index)  # (R, 4096)

        ins_gan_loss = self.instance_level_gan_loss(s_fc7, s_label, t_fc7, t_label, num_class=self.num_class)

        # Original Faster R-CNN training continues
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            s_roi,
            at.tonumpy(s_bbox),
            at.tonumpy(s_label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            s_features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(s_bbox),
            s_anchor,
            s_img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            s_rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)
        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(s_rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(s_rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(s_gt_roi_label).long()]
        s_gt_roi_label = at.totensor(s_gt_roi_label).long()
        s_gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            s_gt_roi_loc,
            s_gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, s_gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), s_gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses), img_gan_loss, ins_gan_loss

    def image_level_gan_loss(self, s_feat, t_feat, s_img_size, t_img_size):
        s_grid_l, s_grid_m, s_grid_s = self.split_pooling((s_img_size[1], s_img_size[0]), s_feat)
        t_grid_l, t_grid_m, t_grid_s = self.split_pooling((t_img_size[1], t_img_size[0]), t_feat)

        grid_pair_l_ss, grid_pair_l_st = pair(s_grid_l, t_grid_l)
        grid_pair_m_ss, grid_pair_m_st = pair(s_grid_m, t_grid_m)
        grid_pair_s_ss, grid_pair_s_st = pair(s_grid_s, t_grid_s)

        criterion = nn.CrossEntropyLoss()
        l_pairs = t.cat((grid_pair_l_ss, grid_pair_l_st))  # (R + R', 7, 7)
        l_labels = t.cat((t.zeros(len(grid_pair_l_ss)), t.ones(len(grid_pair_l_st))))
        m_pairs = t.cat((grid_pair_m_ss, grid_pair_m_st))
        m_labels = t.cat((t.zeros(len(grid_pair_m_ss)), t.ones(len(grid_pair_m_st))))
        s_pairs = t.cat((grid_pair_s_ss, grid_pair_s_st))
        s_labels = t.cat((t.zeros(len(grid_pair_s_ss)), t.ones(len(grid_pair_s_st))))

        out_l = self.gan_il(l_pairs)
        out_m = self.gan_im(m_pairs)
        out_s = self.gan_is(s_pairs)
        loss = criterion(out_l, l_labels)
        loss += criterion(out_m, m_labels)
        loss += criterion(out_s, s_labels)

        return loss

    def instance_level_gan_loss(self, s_fc7, s_label, t_fc7, t_label, num_class):
        ss_pair = list()
        st_pair = list()
        ss_gan_label = list()
        st_gan_label = list()
        for i in range(num_class):
            s_idx = t.where(s_label == i)
            t_idx = t.where(t_label == i)
            if len(s_idx) == 0 or len(t_idx) == 0:
                continue
            ss, st = pair(s_fc7[s_idx], t_fc7[t_idx])  # (l_i, 2 * 4096)
            ss_pair.append(ss)  # R * (l_i, 2 * 4096) --cat--> (R * l_i, 2 * 4096)
            st_pair.append(st)
            ss_gan_label.append(2 * i * t.ones(len(ss)))
            st_gan_label.append(2 * i * t.ones(len(st)) + 1)

        ss_pair = t.cat(ss_pair)
        st_pair = t.cat(st_pair)
        ss_gan_label = t.cat(ss_gan_label).int()
        st_gan_label = t.cat(st_gan_label).int()
        # TODO Decide whether feed all in one time
        # dataloader = instance_pair_dataloader(ss_pair, st_pair, ss_gan_label, st_gan_label)
        pairs = t.cat((ss_pair, st_pair))
        labels = t.cat((ss_gan_label, st_gan_label))
        criterion = nn.CrossEntropyLoss()
        out = self.gan_ins(pairs)
        loss = criterion(out, labels)
        return loss

    def train_step(self, source_imgs, target_imgs, s_bboxes, s_labels, t_bboxes, t_labels, scale):
        self.optimizer.zero_grad()
        self.gan_optim.zero_grad()
        losses, img_gan_loss, ins_gan_loss = self.forward(source_imgs, target_imgs, s_bboxes, s_labels, t_bboxes, t_labels, scale)
        losses.total_loss.backward()
        img_gan_loss.backward()
        ins_gan_loss.backward()
        self.optimizer.step()
        self.gan_optim.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

    def _init_optim(self):
        gan_il_param = self.gan_il.parameters()
        gan_im_param = self.gan_im.parameters()
        gan_is_param = self.gan_is.parameters()
        gan_ins_param = self.gan_ins.parameters()
        gan_optim = t.optim.SGD([{'params': gan_il_param},
                                 {'params': gan_im_param},
                                 {'params': gan_is_param},
                                 {'params': gan_ins_param}], lr=opt.lr, momentum=0.9)
        return gan_optim


def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
