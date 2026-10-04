"""Shared locations. Everything lives inside this repository; nothing is read from outside it."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRAINING = ROOT / "training"
DETECTOR_WEIGHTS = TRAINING / "yolo" / "runs" / "detect" / "train1" / "weights" / "best.pt"
CLASSIFIER_WEIGHTS = TRAINING / "reid" / "models" / "classifier.pt"  # make/model/generation, 9,630 classes
EMBEDDINGS_WEIGHTS = TRAINING / "reid" / "models" / "embeddings_model.pt"  # re-ID embedding, 5,445-class head unused
CLASS_META = TRAINING / "reid" / "models" / "classes_9630.json"
SOURCE_VIDEOS = ROOT / "demo_footage" / "sources"  # raw footage, not committed

if str(TRAINING / "reid") not in sys.path:
    sys.path.append(str(TRAINING / "reid"))  # model definitions: src.models.classifier
