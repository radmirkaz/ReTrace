# ReTrace model training

Everything needed to reproduce the three models ReTrace runs on:

| Model | File | What it does in ReTrace |
|---|---|---|
| Vehicle detector | `yolo/runs/detect/train1/weights/best.pt` | YOLOv12n fine-tuned to one class (`car`); finds every vehicle in every frame, tracked with ByteTrack |
| Classifier | `reid/models/classifier.pt` | Make / model / generation, 9,630 classes |
| Embeddings model | `reid/models/embeddings_model.pt` | 2048-d appearance embedding used for re-identification and look-alike search |

The classifier and the embeddings model share the same code and architecture
(EfficientNetV2-M, 300×300 input, 2048-d embedding, metric learning with contrastive + circle loss plus a
cross-entropy classification head). The embeddings model uses less data, a subset focused on cars
sold in the US and Canada (5,445 classes), with parameters chosen for embedding quality rather than
classification accuracy.

```
training/
├── yolo/
│   ├── data.yaml               # dataset config (1 class: car)
│   ├── preproc.ipynb           # convert / clean / merge source datasets to YOLO format
│   ├── unite.ipynb             # merge the prepared datasets into train/ and valid/
│   ├── train.ipynb             # training
│   ├── infer.ipynb             # inference on images / video
│   ├── speed_tracking.ipynb    # tracking + speed estimation
│   ├── video_processor.py
│   └── runs/detect/train1/     # final run: weights/best.pt, metrics and curves
└── reid/
    ├── src/                    # models (EfficientNetV2, re-ID wrapper), dataset, losses, transforms
    ├── train/train_effnet.py   # training entry point
    ├── scripts/                # dataset preparation (GigaVehicle / Stanford Cars)
    ├── datasets/               # annotations + splits (images are downloaded separately)
    ├── models/
    │   ├── classifier.pt          # 9,630-class classifier
    │   ├── classes_9630.json      # class id → make/model/generation, market and origin
    │   └── embeddings_model.pt    # embeddings model for re-identification (US/Canada-focused)
    ├── logs/                   # TensorBoard logs
    ├── validate_stanford.ipynb # zero-shot embedding validation on Stanford Cars
    ├── generate_database.ipynb # build the embedding / class database
    └── parse_google.ipynb      # scrape images for new / missing models
```

## Setup

Python 3.10 and a CUDA GPU. Re-ID training ran on one RTX 3090 (about 28 h for 8 epochs).

```bash
git lfs install
git clone https://github.com/radmirkaz/retrace.git
cd retrace/training
pip install torch torchvision ultralytics supervision "pytorch-lightning<2" pytorch-metric-learning albumentations opencv-python numpy pandas scipy scikit-learn matplotlib tqdm loguru unidecode pyyaml
```

The weights (`*.pt`) are stored with Git LFS. Run `git lfs pull` if they arrive as small pointer files.
`train_effnet.py` uses PyTorch Lightning 1.x trainer arguments (`gpus`, `auto_lr_find`), hence `pytorch-lightning<2`.

## Datasets

Images are not stored in the repository. Both training sets are published on Kaggle in the layout the code
expects: open the link, press **Download**, and extract the archive where shown below.

### Detector data (~13 GB)

Kaggle: https://tinyurl.com/retrace-yolo-dataset

Combined from these sources and converted to a single `car` class:

| Source | Link |
|---|---|
| Vehicles (Roboflow 100, `vehicles-q0x2v`) | https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2 |
| Stanford Cars (detection boxes) | https://universe.roboflow.com/openglpro/stanford_car/dataset/8 |
| Self-Driving Car (Udacity subset) | https://universe.roboflow.com/roboflow-gw7yv/self-driving-car/dataset/2 |

Extract into `yolo/`:

```
yolo/
├── data.yaml
├── data/          # original per-source datasets (selfdriving, stanford, v2, v4), used by preproc.ipynb
├── train/{images,labels}
└── valid/{images,labels}
```

### Re-ID data (~108 GB, ~407k images)

Kaggle: https://tinyurl.com/retrace-reid-dataset

- **GigaVehicle / GigaFlexhicle**: ~390k cropped vehicle photos split by generation (all bodies and
  restylings of one generation are one class). Original: https://www.kaggle.com/datasets/dmitrygaus/gigaflexhicle
- **Google Images**: newer and popular North American models missing from GigaVehicle, collected with
  `parse_google.ipynb` and [Google-Image-Scraper](https://github.com/ohyicong/Google-Image-Scraper) from the
  list in `reid/datasets/parse.csv`. Corrupted images, interior shots and duplicates were removed.

Extract into `reid/datasets/data/`:

```
reid/datasets/data/
├── images/                       # <brand>/<model>/<generation>/*.jpg + Google-scraped class folders
├── annotation_fixed_addon2.csv   # img_path, class (all 407k images, 9,630 classes)
├── train_full.txt                # train + val
├── train.txt                     # 90 % stratified split
└── val.txt                       # 10 % stratified split
```

The annotation and split files are included. To regenerate the splits: `cd reid/datasets && python create_split.py`.

### Stanford Cars (validation only)

Never used for training. It measures zero-shot embedding quality (`reid/validate_stanford.ipynb`) and is the
validation set inside `train_effnet.py`. Download it with the devkit, e.g.
https://www.kaggle.com/datasets/eduardo4jesus/stanford-cars-dataset, and place it as:

```
reid/datasets/stanford/
├── car_devkit/   # cars_meta.mat, cars_train_annos.mat, cars_test_annos.mat, ...
├── cars_train/
└── cars_test/
```

```bash
cd reid/scripts && python prepare_stanford.py
```

**BRCars-427** (https://github.com/danimtk/brcars-dataset) was evaluated and rejected: ~58k unlabeled interior
images and large overlap with GigaVehicle.

## Training the detector

```bash
cd yolo
jupyter nbconvert --to notebook --execute train.ipynb   # or open it and run all cells
```

The notebook starts from `yolo12n.pt` and trains with `imgsz=720, batch=32, epochs=50, lr0=3e-4`. To rebuild
the dataset from the raw sources, run `preproc.ipynb`, then `unite.ipynb`. Ultralytics writes each run to
`runs/detect/train*/`; the shipped weights are `runs/detect/train1/weights/best.pt`.

## Training the classifier and the embeddings model

Both models come from the same entry point. Settings are constants at the top of `reid/train/train_effnet.py`.

```bash
cd reid/train
python train_effnet.py
tensorboard --logdir ../logs
```

**Classifier (`classifier.pt`)**: run the script as it is. It trains on `train_full.txt` (9,630 classes) for
8 epochs at 300×300, batch 16, AdamW (lr 6e-5, weight decay 0.05, ×0.1 after epoch 4), mixed precision,
gradient accumulation 1, then 2 from epoch 2, then 3 from epoch 4.

**Embeddings model (`embeddings_model.pt`)**: the same script with `num_classes=5445` in the
`CarsReidentificationTrainEffnet(...)` call and the dataset `split` pointed at the US/Canada-focused subset
of the annotations, with parameters tuned for embedding quality instead of classification accuracy.

| Constant | Meaning |
|---|---|
| `DATASET_ROOT_DIR` | folder with `images/` and the split files |
| `PRETRAIN` | optional state dict to initialise from (the classification layer is skipped, so the class count may differ) |
| `MODEL_RESUME` | Lightning checkpoint to resume from |
| `TARGET_SHAPE`, `BATCH_SIZE`, `AUGS` | input size, batch size, augmentations |

Lightning checkpoints are written to `reid/logs/` every 25,000 steps. The files in `reid/models/` are the
state dict of the inner network (`model.net.state_dict()`), which ReTrace loads into
`src.models.classifier.EffNetv2(class_num, features_dim=2048, model_name="m")`.

## Validation

Run from `reid/`:

- `validate_stanford.ipynb`: zero-shot retrieval metrics on Stanford Cars
- `generate_database.ipynb`: embeddings / class metadata database

Re-identification across real traffic cameras is measured by `pipeline/eval_reid.py` in the repository root
(CityFlowV2 S01).

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics), [supervision](https://github.com/roboflow/supervision)
- [pytorch-efficientnet](https://github.com/abhuse/pytorch-efficientnet)
- [Google-Image-Scraper](https://github.com/ohyicong/Google-Image-Scraper) by ohyicong
- The dataset authors listed above
