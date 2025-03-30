import os
import shutil
import random

# Paths
SOURCE_DIR = "../data/garbage_mapped"
TARGET_DIR = "../data"
CLASSES = ["compost", "recycle", "garbage"]

# Split ratios
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

for cls in CLASSES:
    src_folder = os.path.join(SOURCE_DIR, cls)
    images = os.listdir(src_folder)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:]
    }

    for split, files in splits.items():
        split_dir = os.path.join(TARGET_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)

        for img in files:
            src_path = os.path.join(src_folder, img)
            dst_path = os.path.join(split_dir, img)
            shutil.copy2(src_path, dst_path)