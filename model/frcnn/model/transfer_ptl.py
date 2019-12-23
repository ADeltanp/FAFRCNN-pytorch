import numpy as np
from model.frcnn.model.utils.bbox_tools import bbox_iou


class TransferPTL(object):
    """Proposal Target Layer designed for transfer learning

    Args:
        pos_iou_thresh (float): IoU threshold for a RoI to be considered as a
            foreground.
    """

    def __init__(self, pos_iou_thresh=0.7):
        self.pos_iou_thresh = pos_iou_thresh

    def __call__(self, roi, bbox, label,):
        """Assigns ground truth to sampled proposals.
        Here are notations.

        * :math:`S` is the total number of sampled RoIs, which equals \
            :obj:`self.n_sample`.
        * :math:`L` is number of object classes possibly including the \
            background.

        Args:
            roi (array): Region of Interests (RoIs) from which we sample.
                Its shape is :math:`(R, 4)`
            bbox (array): The coordinates of ground truth bounding boxes.
                Its shape is :math:`(R', 4)`.
            label (array): Ground truth bounding box labels. Its shape
                is :math:`(R',)`. Its range is :math:`[0, L - 1]`, where
                :math:`L` is the number of foreground classes.

        Returns:
            (array, array):

            * **sample_roi**: Regions of interests that are sampled. \
                Its shape is :math:`(S, 4)`.
            * **gt_roi_label**: Labels assigned to sampled RoIs. Its shape is \
                :math:`(S,)`. Its range is :math:`[0, L]`. The label with \
                value 0 is the background.

        """
        n_bbox, _ = bbox.shape
        roi = np.concatenate((roi, bbox), axis=0)

        iou = bbox_iou(roi, bbox)
        gt_assignment = iou.argmax(axis=1)
        max_iou = iou.max(axis=1)
        # Offset range of classes from [0, n_fg_class - 1] to [1, n_fg_class].
        # The label with value 0 is the background.
        gt_roi_label = label[gt_assignment] + 1

        # Select foreground RoIs as those with >= pos_iou_thresh IoU.
        pos_index = np.where(max_iou >= self.pos_iou_thresh)[0]

        gt_roi_label = gt_roi_label[pos_index]
        sample_roi = roi[pos_index]

        return sample_roi, gt_roi_label
