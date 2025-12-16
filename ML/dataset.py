# ml/dataset.py

import os
import cv2
import torch
from torch.utils.data import Dataset

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png")

# Crops we want to train on
ALLOWED_CROPS = ("Apple", "Potato", "Tomato")

class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir should contain:
        - color/
        - grayscale/
        - segmented/
        """
        self.root_dir = root_dir
        self.transform = transform

        self.images = []
        self.class_to_idx = {}
        self.classes = []

        self._prepare_dataset()

        print(f"Loaded {len(self.images)} images from {len(self.classes)} classes")

    def _prepare_dataset(self):
        class_set = set()

        # First pass: collect allowed class names
        for variant in ["color", "grayscale", "segmented"]:
            variant_path = os.path.join(self.root_dir, variant)
            if not os.path.isdir(variant_path):
                continue

            for cls in os.listdir(variant_path):
                if cls.startswith(ALLOWED_CROPS):
                    cls_path = os.path.join(variant_path, cls)
                    if os.path.isdir(cls_path):
                        class_set.add(cls)

        self.classes = sorted(class_set)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Second pass: collect images
        for variant in ["color", "grayscale", "segmented"]:
            variant_path = os.path.join(self.root_dir, variant)
            if not os.path.isdir(variant_path):
                continue

            for cls in os.listdir(variant_path):
                if not cls.startswith(ALLOWED_CROPS):
                    continue

                cls_path = os.path.join(variant_path, cls)
                if not os.path.isdir(cls_path):
                    continue

                label = self.class_to_idx[cls]

                for file in os.listdir(cls_path):
                    if file.lower().endswith(VALID_EXTENSIONS):
                        self.images.append(
                            (os.path.join(cls_path, file), label)
                        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Could not read image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]

        return image, torch.tensor(label)


