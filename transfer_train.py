from __future__ import absolute_import
# though cupy is not used but without this line, it raise errors...
import cupy as cp
import os

import ipdb
import matplotlib
from itertools import cycle
from tqdm import tqdm

from model.frcnn.utils.config import opt
from model.frcnn.data.traindataset import TrainDataset, TestDataset, inverse_normalize
from model.frcnn.model.faster_rcnn_vgg16 import FasterRCNNVGG16
from torch.utils import data as data_
from transfer_trainer import TransferTrainer
from model.frcnn.utils import array_tool as at
from model.frcnn.utils.vis_tool import visdom_bbox
from model.frcnn.utils.eval_tool import eval_detection_voc

# # fix for ulimit
# # https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
# import resource
#
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')


def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def transfer_train(faster_rcnn, **kwargs):
    opt._parse(kwargs)

    dataset = TrainDataset(opt)
    print('load data')
    source_dataloader = data_.DataLoader(dataset,
                                         batch_size=1,
                                         shuffle=True,
                                         # pin_memory=True,
                                         num_workers=opt.num_workers)

    target_set = TestDataset(opt, 'val', data_path=opt.dota_data_dir)
    target_dataloader = data_.DataLoader(target_set,
                                         batch_size=1,
                                         num_workers=opt.num_workers,
                                         shuffle=True,
                                         pin_memory=True)

    testset = TestDataset(opt, data_path=opt.dota_data_dir)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False,
                                       pin_memory=True)

    trainer = TransferTrainer(faster_rcnn, opt.dota_num_class).cuda()

    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    target_len = len(target_set)
    for epoch in range(opt.transfer_epoch):
        trainer.reset_meters()
        for ii, (s_img, s_bbox_, s_label_, s_scale,
                 t_img, t_bbox_, t_label_, t_scale
                 ) in tqdm(enumerate(zip(source_dataloader, cycle(target_dataloader)))):
            s_scale = at.scalar(s_scale)
            s_img, s_bbox, s_label = s_img.cuda().float(), s_bbox_.cuda(), s_label_.cuda()
            t_img, t_bbox, t_label = t_img.cuda().float(), t_bbox_.cuda(), t_label_.cuda()
            trainer.train_step(s_img, s_bbox, s_label, s_scale,
                               t_img, t_bbox, t_label, t_scale)
# TODO DP CONTINUE HERE & EVAL
            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(s_img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(s_bbox_[0]),
                                     at.tonumpy(s_label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
