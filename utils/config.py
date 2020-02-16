from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwritten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    DOTA_LABEL_NAMES = (
        'plane',
        'ship',
        'storage-tank',
        'baseball-diamond',
        'tennis-court',
        'basketball-court',
        'ground-track-field',
        'harbor',
        'bridge',
        'large-vehicle',
        'small-vehicle',
        'helicopter',
        'roundabout',
        'soccer-ball-field',
        'swimming-pool',
        'container-crane'
    )
    clear_data_dir = 'E:/Data/NN/DOTA'
    hazy_data_dir = 'E:/Data/NN/hazy_DOTA'
    dota_num_class = len(DOTA_LABEL_NAMES)
    # TODO MOD size of img
    min_size = 900   # image resize
    max_size = 1500  # image resize
    batch_size = 1
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'dota'  # TODO MOD preset?
    pretrained_model = 'vgg16'

    # training
    epoch = 14
    transfer_epoch = 14

    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()