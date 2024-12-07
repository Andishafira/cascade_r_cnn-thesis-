from set_lib_dir import LIB_ROOT_DIR
import os
_base_ = './centernet_resnet18_140e_coco.py'
data_root = LIB_ROOT_DIR + '/data/'


data = dict(
    train=dict(
        ann_file=data_root + 'train/annotation/train.json',
        img_prefix=data_root + 'train/images/',
    ),
    val=dict(
        ann_file=data_root + 'valid/annotation/val.json',
        img_prefix=data_root + 'valid/images/',
    ),
    test=dict(
        ann_file=data_root + 'test/annotation/test.json',
        img_prefix=data_root + 'test/images/',
    )
)

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,
    warmup_ratio=1.0 / 1000,
    step=[15, 18])
runner = dict(max_epochs=20)
evaluation = dict(interval=20, metric='bbox')
load_from = LIB_ROOT_DIR + '/work_dirs/centernet_resnet18_140e_coco/latest.pth'
