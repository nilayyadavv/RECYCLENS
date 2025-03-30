import os
import shutil


class_map = {
    "biological": "compost",
    "cardboard": "compost",
    "paper": "recycle",
    "plastic": "recycle",
    "metal": "recycle",
    "brown-glass": "recycle",
    "green-glass": "recycle",
    "white-glass": "recycle",
    "trash": "garbage",
    "battery": "garbage",
    "clothes": "garbage",
    "shoes": "garbage"
}

source_folder = "../data/garbage_classification"
target_folder = "../data/garbage_mapped"

for target_class in ["compost", "recycle", "garbage"]:
    os.makedirs(os.path.join(target_folder, target_class), exist_ok=True)

# Move files
for original_class, mapped_class in class_map.items():
    src_path = os.path.join(source_folder, original_class)
    dst_path = os.path.join(target_folder, mapped_class)

    if os.path.isdir(src_path):
        for file in os.listdir(src_path):
            full_file_path = os.path.join(src_path, file)
            if os.path.isfile(full_file_path):
                shutil.copy2(full_file_path, dst_path)