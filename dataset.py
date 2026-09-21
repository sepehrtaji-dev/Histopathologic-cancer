
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class HistopathDataset(Dataset):
    def __init__(
        self,
        data_dir="data",
        csv_file="data/_labels.csv",
        image_size=96,
        training=True
    ):
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_file)

        columns = {
            column.lower(): column
            for column in self.df.columns
        }

        id_col = None
        label_col = None

        for name in ["id", "image_id", "image", "filename"]:
            if name in columns:
                id_col = columns[name]
                break

        for name in ["label", "target", "cancer"]:
            if name in columns:
                label_col = columns[name]
                break

        if id_col is None or label_col is None:
            raise ValueError(
                f"Could not find image ID and label columns.\n"
                f"CSV columns: {list(self.df.columns)}"
            )

        self.id_col = id_col
        self.label_col = label_col

        if training:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.15
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def _find_image(self, image_id):
        image_id = str(image_id)

        candidates = [
            self.data_dir / image_id,
            self.data_dir / f"{image_id}.tif",
            self.data_dir / f"{image_id}.tiff",
            self.data_dir / f"{image_id}.png",
            self.data_dir / f"{image_id}.jpg",
            self.data_dir / f"{image_id}.jpeg"
        ]

        for path in candidates:
            if path.exists():
                return path

        raise FileNotFoundError(
            f"Image not found for ID: {image_id}"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_id = row[self.id_col]
        label = float(row[self.label_col])

        image_path = self._find_image(image_id)

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.float32
        )

