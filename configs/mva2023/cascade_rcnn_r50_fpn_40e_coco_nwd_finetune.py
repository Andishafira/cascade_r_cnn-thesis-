from set_lib_dir import LIB_ROOT_DIR
_base_ = '/content/cascade_r_cnn-thesis-/config/mva2023/cascade_rcnn_r50_fpn_140e_coco_nwd.py'
data_root = '/content/cascade_r_cnn-thesis-/data/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                meta_keys=('filename', 'ori_shape', 'img_shape',
                           'scale_factor', 'flip', 'flip_direction',
                           'img_norm_cfg'),
                keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=16,
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
        pipeline=test_pipeline
    )
)        
runner = dict(max_epochs=40)

load_from = LIB_ROOT_DIR + '/content/cascade_r_cnn-thesis-/work_dirs/cascade_rcnn_r50_fpn_140e_coco_nwd/latest.pth'

log_config = dict(
    interval=100,
)