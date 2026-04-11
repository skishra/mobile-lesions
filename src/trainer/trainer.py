import torch
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        optimizer=None,
        device=None,
        scheduler=None,
        threshold=0.5,
        save_path="best_model.pt"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.threshold = threshold
        self.save_path = save_path

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_loss = float("inf")

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0

        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            loop.set_postfix(loss=loss.item())

        return total_loss / len(self.train_loader)

    def validate(self):
        if self.val_loader is None:
            return None

        self.model.eval()
        total_loss = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validation", leave=False)

            for images, labels in loop:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()

                probs = torch.sigmoid(outputs)
                preds = (probs > self.threshold).float()

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

                loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.val_loader)

        return avg_loss

    def fit(self, epochs):
        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")

            train_loss = self.train_one_epoch()
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = self.validate()
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    torch.save(self.model, self.save_path)
                    print("Saved best model")

            if self.scheduler:
                self.scheduler.step()

    def test_roc_auc(self, model, test_loader):
        model = model.to(self.device)
        model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            loop = tqdm(test_loader, desc="Testing ROC AUC")

            for images, labels in loop:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                probs = torch.sigmoid(outputs)

                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs).numpy()
        all_labels = torch.cat(all_labels).numpy()

        try:
            if all_labels.ndim == 1 or all_labels.shape[1] == 1:
                auc = roc_auc_score(all_labels, all_probs)
            else:
                auc = roc_auc_score(all_labels, all_probs, average="macro", multi_class="ovr")
        except ValueError as e:
            print("ROC AUC could not be computed:", e)
            return None

        print(f"\nTest ROC AUC: {auc:.4f}")
        return auc


if __name__ == "__main__":
    from pathlib import Path
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.dataset.dataset import ISIC2019Dataset

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    train_csv = Path("src/dataset/after_split/train.csv")
    val_csv = Path("src/dataset/after_split/val.csv")
    test_csv = Path("src/dataset/after_split/test.csv")
    image_dir = Path("src/dataset/lesions-kaggle/ISIC_2019_Training_Input/")

    train_dataset = ISIC2019Dataset(
        csv_file=train_csv,
        image_dir=image_dir,
        transform=transform
    )

    val_dataset = ISIC2019Dataset(
        csv_file=val_csv,
        image_dir=image_dir,
        transform=transform
    )

    test_dataset = ISIC2019Dataset(
        csv_file=test_csv,
        image_dir=image_dir,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    test_loader = DataLoader(test_dataset, batch_size=32)

    model = timm.create_model(
        'mobilenetv4_conv_medium.e500_r256_in1k',
        pretrained=True,
        num_classes=len(train_dataset.classes)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
    )

    # trainer.fit(epochs=10)

    model = torch.load("best_model.pt", map_location="cpu")

    trainer.test_roc_auc(model, test_loader)
