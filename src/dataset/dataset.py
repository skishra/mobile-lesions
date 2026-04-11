import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset


class ISIC2019Dataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_dir,
        transform=None,
        img_ext=".jpg",
        return_id=False
    ):
        """
        Args:
            csv_file (str): Path to CSV file
            image_dir (str): Directory with images
            transform (callable, optional): Transform to apply
            img_ext (str or None): Force extension ('.jpg', '.png') or None to auto-detect
            return_id (bool): Whether to return image ID
        """
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.img_ext = img_ext
        self.return_id = return_id

        # Class columns (everything except 'image')
        self.classes = self.df.columns[1:].tolist()

    def __len__(self):
        return len(self.df)

    def _get_image_path(self, image_id):
        if self.img_ext is not None:
            return os.path.join(self.image_dir, image_id + self.img_ext)

        # Auto-detect extension
        for ext in [".jpg", ".png", ".jpeg"]:
            path = os.path.join(self.image_dir, image_id + ext)
            if os.path.exists(path):
                return path

        raise FileNotFoundError(f"Image not found for ID: {image_id}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_id = row["image"]
        img_path = self._get_image_path(image_id)

        image = Image.open(img_path).convert("RGB")

        # Multi-label target (float tensor)
        label = torch.tensor(row[1:].values.astype("float32"))

        if self.transform:
            image = self.transform(image)

        if self.return_id:
            return image, label, image_id

        return image, label


if __name__ == "__main__":
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

    test_loader = DataLoader(test_dataset, batch_size=32)
    
    print(type(test_loader))
