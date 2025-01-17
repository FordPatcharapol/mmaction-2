from mmengine import Config
import os.path as osp
import mmengine
from mmengine.runner import Runner

cfg = Config.fromfile(
    './configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py')

# Modify dataset type and path
cfg.data_root = './data/lotus/train/'
cfg.data_root_val = './data/lotus/val/'
cfg.ann_file_train = './data/lotus/lotus_train_video.txt'
cfg.ann_file_val = './data/lotus/lotus_val_video.txt'

cfg.test_dataloader.dataset.ann_file = './data/lotus/lotus_val_video.txt'
cfg.test_dataloader.dataset.data_prefix.video = './data/lotus/val/'

cfg.train_dataloader.dataset.ann_file = './data/lotus/lotus_train_video.txt'
cfg.train_dataloader.dataset.data_prefix.video = './data/lotus/train/'

cfg.val_dataloader.dataset.ann_file = './data/lotus/lotus_val_video.txt'
cfg.val_dataloader.dataset.data_prefix.video = './data/lotus/val/'

# Modify num classes of the model in cls_head
cfg.model.cls_head.num_classes = 2
# We can use the pre-trained TSN model
cfg.load_from = 'tsn_imagenet-pretrained-r50_8xb32-1x1x8-100e_kinetics400-rgb_20220906-2692d16c.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './tutorial_exps'

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.train_dataloader.batch_size = 4
cfg.val_dataloader.batch_size = 4
cfg.test_dataloader.batch_size = 4
cfg.optim_wrapper.optimizer.lr = cfg.optim_wrapper.optimizer.lr / 8 / 16
cfg.train_cfg.max_epochs = 10

cfg.train_dataloader.num_workers = 1
cfg.val_dataloader.num_workers = 1
cfg.test_dataloader.num_workers = 1
# cfg.train_dataloader.persistent_workers = False
# cfg.val_dataloader.persistent_workers = False
# cfg.test_dataloader.persistent_workers = False

# Create work_dir
mmengine.mkdir_or_exist(osp.abspath(cfg.work_dir))

# build the runner from config
runner = Runner.from_cfg(cfg)

# start training
runner.train()
