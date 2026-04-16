import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ISIC2019Dataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None, img_ext=".jpg"):
        self.df = pd.read_csv(csv_file)
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.img_ext = img_ext

        # Class columns (everything except 'image')
        self.classes = self.df.columns[1:].tolist()

        if "image" not in self.df.columns:
            raise ValueError(
                f"CSV must have an 'image' column. Found: {self.df.columns.tolist()}")

        self.label_columns = [c for c in self.df.columns if c != "image"]

        # Pre-resolve paths when auto-detecting extension
        if self.img_ext is None:
            self._path_cache = {
                img_id: self._resolve_path(img_id)
                for img_id in self.df["image"]
            }

    def _resolve_path(self, image_id):
        for ext in (".jpg", ".png", ".jpeg"):
            path = self.image_dir / (image_id + ext)
            if path.exists():
                return path
        raise FileNotFoundError(f"No image found for ID: {image_id}")

    def _get_image_path(self, image_id):
        if self.img_ext is not None:
            return self.image_dir / (image_id + self.img_ext)
        return self._path_cache[image_id]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image"]

        with Image.open(self._get_image_path(image_id)) as img:
            image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)

        label = torch.from_numpy(
            row[self.label_columns].to_numpy(dtype="float32")
        )

        return image, label


def main():
    from torchvision import transforms
    from torch.utils.data import DataLoader

    csv_file = Path("src/dataset/after_split/test.csv")
    image_dir = Path("src/dataset/lesions-kaggle/ISIC_2019_Training_Input/")

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    test_dataset = ISIC2019Dataset(
        csv_file=csv_file,
        image_dir=image_dir,
        transform=transform
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )

    print(type(test_loader))


if __name__ == "__main__":
    main()
