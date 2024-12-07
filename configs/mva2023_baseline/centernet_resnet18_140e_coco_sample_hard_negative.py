_base_ = './centernet_resnet18_140e_coco.py'
data_root = 'data/'

data = dict(
    test=dict(
        samples_per_gpu=4,
        ann_file=data_root + 'train/annotation/train.json',
        img_prefix=data_root + 'train/images/',
    ) 
)
