import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl

sys.path.append("..")
import src.models.carsreid
import src.datasets.CarsDataset
import src.utils.transforms
import src.utils.vis

import warnings

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    MODEL_RESUME = None
    PRETRAIN = None
    DATASET_ROOT_DIR = "../datasets/data"
    MODELS_DIR = "../models"
    LOGS_DIR = "../logs"

    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]
    TARGET_SHAPE = (300, 300)
    # TARGET_SHAPE = (384, 384)
    BATCH_SIZE = 16
    AUGS = True

    model = src.models.carsreid.CarsReidentificationTrainEffnet(num_classes=9630, pretain_path=PRETRAIN, model_name="m")

    train_transforms = src.utils.transforms.CarsTransforms(
        TARGET_SHAPE, IMAGE_MEAN, IMAGE_STD, augs=AUGS
    )
    trainset = ConcatDataset(
        [
            src.datasets.CarsDataset.CarsDataset(
                DATASET_ROOT_DIR, split="/train_full", transforms=train_transforms, triplet=True, augs=AUGS
            ),
            # src.datasets.CarsDataset.StanfordCarsDataset(
            #     DATASET_ROOT_DIR, split="train", transforms=train_transforms, triplet=True, augs=True
            # ),
        ]
    )

    num_classes = trainset.datasets[0].CLASS_NUM
    print("Num classes: ", num_classes)

    test_transforms = src.utils.transforms.CarsTransforms(
        TARGET_SHAPE, IMAGE_MEAN, IMAGE_STD, augs=False
    )
    testset = src.datasets.CarsDataset.StanfordCarsDataset(
        DATASET_ROOT_DIR, split="train", transforms=test_transforms
    )

    train_loader = DataLoader(
        dataset=trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        dataset=testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )


    trainer_cfg = {
        "gpus": 1,
        "logger": pl.loggers.TensorBoardLogger(save_dir=LOGS_DIR),
        "precision": 16,
        "auto_lr_find": True,
        "accumulate_grad_batches": {0: 1, 2: 2, 4: 3},
        "max_epochs": 8,
        "callbacks": [
            pl.callbacks.ModelCheckpoint(dirpath=LOGS_DIR, every_n_train_steps=25000, save_top_k=-1),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
        ],
    }

    trainer = pl.Trainer(**trainer_cfg)
    _ = trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=MODEL_RESUME,
    )
