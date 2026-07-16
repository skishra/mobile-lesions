import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import torch.nn as nn
from tqdm import tqdm
from src.training.evaluator import Evaluator


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        optimizer=None,
        device=None,
        scheduler=None,
        save_path="best_model.pt",
        patience=5,
        max_grad_norm=1.0,
        class_names=None,
        class_weight=None,
        max_class_weight=None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_path = save_path
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.class_names = class_names
        self.max_class_weight = max_class_weight

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if isinstance(class_weight, str):
            if class_weight not in ("auto", "sqrt", "log1p"):
                raise ValueError(
                    f"class_weight string must be 'auto', 'sqrt', or 'log1p', got {class_weight!r}"
                )
            mode = "raw" if class_weight == "auto" else class_weight
            class_weight = self._compute_class_weight(
                train_loader, mode=mode, max_weight=max_class_weight,
            )
            print(
                f"  Auto class_weight ({mode}, max={max_class_weight}): {class_weight}")

        if class_weight is not None:
            class_weight = class_weight.to(self.device)

        self.criterion = nn.CrossEntropyLoss(weight=class_weight)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long()

            self.optimizer.zero_grad(set_to_none=True)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()

            if self.max_grad_norm:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

    @staticmethod
    def _compute_class_weight(loader, mode="raw", max_weight=None):
        """Compute class_weight = total_samples / (num_classes * class_samples) 
        to handle single-label multi-class imbalance.
        """

        first_batch_labels = next(iter(loader))[1]
        if first_batch_labels.ndim > 1:
            num_classes = first_batch_labels.shape[1]
        else:
            num_classes = 8

        class_counts = torch.zeros(num_classes, dtype=torch.float32)

        for _, labels in loader:
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            for c in range(num_classes):
                class_counts[c] += (labels == c).sum().item()

        total_samples = class_counts.sum()
        class_weight = total_samples / (num_classes * class_counts.clamp(min=1.0))

        if mode == "sqrt":
            class_weight = torch.sqrt(class_weight)
        elif mode == "log1p":
            class_weight = torch.log1p(class_weight)
        elif mode != "raw":
            raise ValueError(
                f"Unknown class_weight mode: {mode!r}. Use 'raw', 'sqrt', or 'log1p'.")

        if max_weight is not None:
            class_weight = class_weight.clamp(max=max_weight)

        return class_weight

    @torch.no_grad()
    def evaluate(self, loader, desc="Validation"):
        """Run evaluation on any loader. Returns loss, AUC, and classification report."""
        self.model.eval()
        total_loss = 0.0

        all_probs = []
        all_labels = []

        loop = tqdm(loader, desc=desc, leave=False)

        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            labels = labels.long()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        try:
            auc = roc_auc_score(all_labels, all_probs,
                                average="macro", multi_class="ovr")
        except ValueError:
            auc = 0.0  

        preds = np.argmax(all_probs, axis=1)
        
        report = classification_report(
            all_labels,
            preds,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True,
        )

        return avg_loss, auc, report

    def _step_scheduler(self, val_loss):
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def fit(self, epochs):
        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(f"\nEpoch [{epoch + 1}/{epochs}]  lr={current_lr:.2e}")

            train_loss = self.train_one_epoch()
            self.history["train_loss"].append(train_loss)
            print(f"  Train Loss: {train_loss:.4f}")

            if self.val_loader is not None:
                val_loss, val_auc, val_report = self.evaluate(self.val_loader)
                self.history["val_loss"].append(val_loss)
                self.history["val_auc"].append(val_auc)
                print(
                    f"  Val Loss:   {val_loss:.4f}  |  Val AUC: {val_auc:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.epochs_without_improvement = 0
                    torch.save(self.model.state_dict(), self.save_path)
                    print("  Saved best model")
                else:
                    self.epochs_without_improvement += 1
                    print(
                        f"  No improvement for {self.epochs_without_improvement}/{self.patience} epochs"
                    )

                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                self._step_scheduler(val_loss)

        if self.val_loader is not None:
            self.model.load_state_dict(torch.load(
                self.save_path, map_location=self.device))
            print(f"\nLoaded best model (val_loss={self.best_val_loss:.4f})")

        return self.history

    def test(self, test_loader):
        self.model.load_state_dict(torch.load(
            self.save_path, map_location=self.device))
        
        evaluator = Evaluator(
            self.model, test_loader,
            device=self.device,
            class_names=self.class_names,
        )
        results = evaluator.evaluate()
        print(
            f"\nTest Loss: {results['loss']:.4f}  |  Test AUC: {results['macro_auc']:.4f}")
        for cls, metrics in results["report"].items():
            if isinstance(metrics, dict):
                print(f"  {cls:>20s}  precision={metrics['precision']:.3f}  "
                      f"recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}")
        return results


def main():
    from pathlib import Path
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.dataset.dataset import ISIC2019Dataset

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    train_csv = Path("src/dataset/after_split/train.csv")
    val_csv = Path("src/dataset/after_split/val.csv")
    test_csv = Path("src/dataset/after_split/test.csv")
    image_dir = Path("src/dataset/archive/ISIC_2019_Training_Input/ISIC_2019_Training_Input/")

    train_dataset = ISIC2019Dataset(
        csv_file=train_csv, image_dir=image_dir, transform=train_transform)
    val_dataset = ISIC2019Dataset(
        csv_file=val_csv, image_dir=image_dir, transform=val_transform)
    test_dataset = ISIC2019Dataset(
        csv_file=test_csv, image_dir=image_dir, transform=val_transform)

    NUM_WORKERS = 4
    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=64,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    model = timm.create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k",
        pretrained=True,
        num_classes=len(train_dataset.classes),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        patience=7,
        class_names=train_dataset.classes,
        class_weight="log1p",
        max_class_weight=3.0,
    )

    history = trainer.fit(epochs=20)
    trainer.test(test_loader)


if __name__ == "__main__":
    main()