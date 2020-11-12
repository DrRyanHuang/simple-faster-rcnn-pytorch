from pprint import pprint


# 此文件为用于 training 的默认配置文件
# 注意 ：配置项可以被通过命令行传进来的参数覆盖
# 例如 ：--voc-data-dir='./data/'

class Config:
    
    # data
    voc_data_dir = '~/Dataset/VOC2007/'
    min_size = 600  # image resize
    max_size = 1000 # image resize
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
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = False # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None

    caffe_pretrain = False # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'


    def _parse(self, kwargs):

        # 读取所有的配置属性
        state_dict = self._state_dict()
        # 迭代新传入的配置, 进行更新操作
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')


    def _state_dict(self):
        # 该方法的实际作用是剔除 `magic_function`

        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}
        # `Config.__dict__` 可以理解一个普通字典，包括之前定义的类属性 


opt = Config()
