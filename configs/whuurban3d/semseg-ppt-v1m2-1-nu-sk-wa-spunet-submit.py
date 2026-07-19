_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
batch_size = 12  # bs: total bs in all gpus
num_worker = 24
mix_prob = 0.8
empty_cache = False
enable_amp = True
find_unused_parameters = True
evaluate = False

# trainer
train = dict(
    type="DefaultTrainer",  # Changed from MultiDatasetTrainer as we isolate to WHU
)

# dataset settings
dataset_type = "WHUDataset"
data_root = "data/whu"
num_classes = 7

# model settings
model = dict(
    type="PPT-v1m2",
    backbone=dict(
        type="SpUNet-v1m3",
        in_channels=6,  # 3 for XYZ coordinates + 3 for RGB color channels
        num_classes=0,
        base_channels=32,
        context_channels=256,
        channels=(32, 64, 128, 256, 256, 128, 96, 96),
        layers=(2, 3, 4, 6, 2, 2, 2, 2),
        enc_mode=False,
        conditions=("WHU",),  # Isolated condition token for your dataset
        zero_init=False,
        norm_decouple=True,
        norm_adaptive=False,
        norm_affine=True,
    ),
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=96,
    context_channels=256,
    conditions=("WHU",),
    num_classes=(num_classes,),  # Tuple mapping matching target classes
)

# scheduler settings
epoch = 50
eval_epoch = 50
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

data = dict(
    num_classes=num_classes,
    ignore_index=-1,
    names=[
        "ground",
        "vegetation",
        "building",
        "wall",
        "bridge",
        "vehicle",
        "barrier",
    ],
    train=dict(
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(
                type="RandomRotate",
                angle=[-1, 1],
                axis="z",
                center=[0, 0, 0],
                p=0.5,
            ),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="train",
                return_grid_coord=True,
            ),
            dict(type="Update", keys_dict={"condition": "WHU"}),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "grid_coord", "segment", "condition"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
        ignore_index=-1,
        loop=1,
    ),
    test=dict(
        type=dataset_type,
        split="val",  # Used validation split for tracking model progression
        data_root=data_root,
        transform=[
            dict(type="Copy", keys_dict={"segment": "origin_segment"}),
            dict(
                type="GridSample",
                grid_size=0.025,
                hash_type="fnv",
                mode="train",
                return_inverse=True,
            ),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=0.05,
                hash_type="fnv",
                mode="test",
                return_grid_coord=True,
            ),
            crop=None,
            post_transform=[
                dict(type="Update", keys_dict={"condition": "WHU"}),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "grid_coord", "index", "condition"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1, 1]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
        ignore_index=-1,
    ),
)



