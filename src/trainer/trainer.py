import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from src.evaluator.evaluator import Evaluator


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
        save_path="best_model.pt",
        patience=5,
        use_amp=True,
        max_grad_norm=1.0,
        class_names=None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.threshold = threshold
        self.save_path = save_path
        self.patience = patience
        self.max_grad_norm = max_grad_norm
        self.class_names = class_names

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

        # Mixed precision
        self.use_amp = use_amp and self.device == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=self.use_amp)

        # History for plotting/logging
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def train_one_epoch(self):
        self.model.train()
        total_loss = 0.0

        loop = tqdm(self.train_loader, desc="Training", leave=False)

        for images, labels in loop:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()

            if self.max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / len(self.train_loader)

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

            with autocast(device_type="cuda", enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item()

            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(loader)

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # Macro AUC across all classes
        try:
            auc = roc_auc_score(all_labels, all_probs,
                                average="macro", multi_class="ovr")
        except ValueError:
            auc = 0.0  # Can happen if a class has no positive samples in the split

        # Per-class predictions for classification report
        preds = (all_probs > self.threshold).astype(int)
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
        # ReduceLROnPlateau needs a metric; everything else takes no args
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
                    print("  ✓ Saved best model")
                else:
                    self.epochs_without_improvement += 1
                    print(
                        f"  No improvement for {self.epochs_without_improvement}/{self.patience} epochs"
                    )

                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping at epoch {epoch + 1}")
                    break

                self._step_scheduler(val_loss)

        # Reload best weights at end of training
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
            device=self.device, threshold=self.threshold,
            class_names=self.class_names, use_amp=self.use_amp,
        )
        results = evaluator.evaluate()
        print(
            f"\nTest Loss: {results['loss']:.4f}  |  Test AUC: {results['macro_auc']:.4f}")
        for cls, metrics in results["report"].items():
            if isinstance(metrics, dict):
                print(f"  {cls:>20s}  precision={metrics['precision']:.3f}  "
                      f"recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}")
        return results


# ──────────────────────────────────────────────────────────────────────
# Example usage
# ──────────────────────────────────────────────────────────────────────

def main():
    from pathlib import Path
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.dataset.dataset import ISIC2019Dataset

    # ImageNet normalization (required by pretrained backbone)
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
    image_dir = Path("src/dataset/lesions-kaggle/ISIC_2019_Training_Input/")

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
    )

    history = trainer.fit(epochs=20)

    # Final test evaluation
    trainer.test(test_loader)


if __name__ == "__main__":
    main()


"""
Output:

Epoch [1/20]  lr=1.00e-04
  Train Loss: 0.3721                                                                                                                                                                                                                                                      
  Val Loss:   0.1904  |  Val AUC: 0.8866                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [2/20]  lr=9.94e-05
  Train Loss: 0.1745                                                                                                                                                                                                                                                      
  Val Loss:   0.1788  |  Val AUC: 0.9203                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [3/20]  lr=9.76e-05
  Train Loss: 0.1521                                                                                                                                                                                                                                                      
  Val Loss:   0.1498  |  Val AUC: 0.9392                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [4/20]  lr=9.46e-05
  Train Loss: 0.1307                                                                                                                                                                                                                                                      
  Val Loss:   0.1593  |  Val AUC: 0.9295                                                                                                                                                                                                                                  
  No improvement for 1/7 epochs

Epoch [5/20]  lr=9.05e-05
  Train Loss: 0.1182                                                                                                                                                                                                                                                      
  Val Loss:   0.1524  |  Val AUC: 0.9396                                                                                                                                                                                                                                  
  No improvement for 2/7 epochs

Epoch [6/20]  lr=8.54e-05
  Train Loss: 0.1092                                                                                                                                                                                                                                                      
  Val Loss:   0.1682  |  Val AUC: 0.9355                                                                                                                                                                                                                                  
  No improvement for 3/7 epochs

Epoch [7/20]  lr=7.94e-05
  Train Loss: 0.0936                                                                                                                                                                                                                                                      
  Val Loss:   0.1456  |  Val AUC: 0.9501                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [8/20]  lr=7.27e-05
  Train Loss: 0.0799                                                                                                                                                                                                                                                      
  Val Loss:   0.1296  |  Val AUC: 0.9530                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [9/20]  lr=6.55e-05
  Train Loss: 0.0700                                                                                                                                                                                                                                                      
  Val Loss:   0.1306  |  Val AUC: 0.9593                                                                                                                                                                                                                                  
  No improvement for 1/7 epochs

Epoch [10/20]  lr=5.78e-05
  Train Loss: 0.0608                                                                                                                                                                                                                                                      
  Val Loss:   0.1464  |  Val AUC: 0.9528                                                                                                                                                                                                                                  
  No improvement for 2/7 epochs

Epoch [11/20]  lr=5.00e-05
  Train Loss: 0.0495                                                                                                                                                                                                                                                      
  Val Loss:   0.1356  |  Val AUC: 0.9510                                                                                                                                                                                                                                  
  No improvement for 3/7 epochs

Epoch [12/20]  lr=4.22e-05
  Train Loss: 0.0426                                                                                                                                                                                                                                                      
  Val Loss:   0.1551  |  Val AUC: 0.9539                                                                                                                                                                                                                                  
  No improvement for 4/7 epochs

Epoch [13/20]  lr=3.45e-05
  Train Loss: 0.0335                                                                                                                                                                                                                                                      
  Val Loss:   0.1755  |  Val AUC: 0.9469                                                                                                                                                                                                                                  
  No improvement for 5/7 epochs

Epoch [14/20]  lr=2.73e-05
  Train Loss: 0.0322                                                                                                                                                                                                                                                      
  Val Loss:   0.1768  |  Val AUC: 0.9458                                                                                                                                                                                                                                  
  No improvement for 6/7 epochs

Epoch [15/20]  lr=2.06e-05
  Train Loss: 0.0250                                                                                                                                                                                                                                                      
  Val Loss:   0.1783  |  Val AUC: 0.9483                                                                                                                                                                                                                                  
  No improvement for 7/7 epochs

Early stopping at epoch 15

Loaded best model (val_loss=0.1296)
                                                                                                                                                                                                                                                                          
Test Loss: 0.1401  |  Test AUC: 0.9517
                   MEL  precision=0.870  recall=0.457  f1=0.599
                    NV  precision=0.834  recall=0.933  f1=0.881
                   BCC  precision=0.837  recall=0.759  f1=0.796
                    AK  precision=0.806  recall=0.287  f1=0.424
                   BKL  precision=0.831  recall=0.450  f1=0.584
                    DF  precision=0.846  recall=0.458  f1=0.595
                  VASC  precision=0.800  recall=0.320  f1=0.457
                   SCC  precision=0.704  recall=0.302  f1=0.422
             micro avg  precision=0.836  recall=0.727  f1=0.778
             macro avg  precision=0.816  recall=0.496  f1=0.595
          weighted avg  precision=0.836  recall=0.727  f1=0.755
           samples avg  precision=0.724  recall=0.727  f1=0.725
"""