import os
import shutil
import yaml
from torch.utils.data import random_split
from tqdm import tqdm

# Get the path to the root directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
config_path = os.path.join(project_root, "config.yaml")


# Load configuration
with open(config_path, "r") as file:
    config = yaml.safe_load(file)

INPUT_DIR = config["data"]["input_dir"]
OUTPUT_DIR = config["data"]["output_dir"]
SPLIT_RATIOS = config["data"]["split_ratios"]

def create_directories(base_dir, categories):
    """Create directories for train/val/test splits and subcategories."""
    for split in SPLIT_RATIOS.keys():
        for category in categories:
            os.makedirs(os.path.join(base_dir, split, category), exist_ok=True)

def split_data(input_dir, output_dir, split_ratios):
    """Split images into train/val/test sets using PyTorch's random_split."""
    categories = os.listdir(input_dir)
    create_directories(output_dir, categories)

    for category in categories:
        category_path = os.path.join(input_dir, category)
        images = os.listdir(category_path)

        # Create indices for each split
        total_images = len(images)
        train_size = int(split_ratios["train"] * total_images)
        val_size = int(split_ratios["val"] * total_images)
        test_size = total_images - train_size - val_size

        splits = random_split(images, [train_size, val_size, test_size])

        # Move images to the respective directories
        for split, split_name in zip(splits, ["train", "val", "test"]):
            for img_index in tqdm(split.indices, desc=f"Processing {split_name} for {category}", unit="image"):
                img = images[img_index]
                src_path = os.path.join(category_path, img)
                dest_path = os.path.join(output_dir, split_name, category, img)
                shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    split_data(INPUT_DIR, OUTPUT_DIR, SPLIT_RATIOS)
    print(f"Data processed and organized in {OUTPUT_DIR}")
