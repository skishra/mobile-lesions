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
        threshold=0.5,
        save_path="best_model.pt",
        patience=5,
        max_grad_norm=1.0,
        class_names=None,
        pos_weight=None,
        max_pos_weight=None
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
        self.max_pos_weight = max_pos_weight

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # --- pos_weight handling ---
        if isinstance(pos_weight, str):
            if pos_weight not in ("auto", "sqrt", "log1p"):
                raise ValueError(
                    f"pos_weight string must be 'auto', 'sqrt', or 'log1p', got {pos_weight!r}"
                )
            mode = "raw" if pos_weight == "auto" else pos_weight
            pos_weight = self._compute_pos_weight(
                train_loader, mode=mode, max_weight=max_pos_weight,
            )
            print(
                f"  Auto pos_weight ({mode}, max={max_pos_weight}): {pos_weight}")

        if pos_weight is not None:
            pos_weight = pos_weight.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self.best_val_loss = float("inf")
        self.epochs_without_improvement = 0

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
    def _compute_pos_weight(loader, mode="raw", max_weight=None):
        """Compute pos_weight = num_negative / num_positive for each class.

        Args:
            loader: Training DataLoader.
            mode: Dampening strategy — "raw", "sqrt", or "log1p".
            max_weight: If set, clamp all weights to this ceiling.
        """
        total_positive = None
        total_samples = 0

        for _, labels in loader:
            if total_positive is None:
                total_positive = labels.sum(dim=0)
            else:
                total_positive += labels.sum(dim=0)
            total_samples += labels.shape[0]

        total_negative = total_samples - total_positive
        pos_weight = total_negative / total_positive.clamp(min=1.0)

        if mode == "sqrt":
            pos_weight = torch.sqrt(pos_weight)
        elif mode == "log1p":
            pos_weight = torch.log1p(pos_weight)
        elif mode != "raw":
            raise ValueError(
                f"Unknown pos_weight mode: {mode!r}. Use 'raw', 'sqrt', or 'log1p'.")

        if max_weight is not None:
            pos_weight = pos_weight.clamp(max=max_weight)

        return pos_weight

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

    # timm 320 test configuration
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
        pos_weight="log1p",
        max_pos_weight=3.0,
    )

    history = trainer.fit(epochs=20)

    # Final test evaluation
    trainer.test(test_loader)


if __name__ == "__main__":
    main()


"""
Output:

Epoch [1/20]  lr=1.00e-04
  Train Loss: 0.3292                                                                                                                                                                                                                                                      
  Val Loss:   0.1836  |  Val AUC: 0.9010                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [2/20]  lr=9.94e-05
  Train Loss: 0.1719                                                                                                                                                                                                                                                      
  Val Loss:   0.1689  |  Val AUC: 0.9227                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [3/20]  lr=9.76e-05
  Train Loss: 0.1516                                                                                                                                                                                                                                                      
  Val Loss:   0.1523  |  Val AUC: 0.9389                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [4/20]  lr=9.46e-05
  Train Loss: 0.1349                                                                                                                                                                                                                                                      
  Val Loss:   0.1376  |  Val AUC: 0.9524                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [5/20]  lr=9.05e-05
  Train Loss: 0.1207                                                                                                                                                                                                                                                      
  Val Loss:   0.1322  |  Val AUC: 0.9565                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [6/20]  lr=8.54e-05
  Train Loss: 0.1066                                                                                                                                                                                                                                                      
  Val Loss:   0.1283  |  Val AUC: 0.9577                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [7/20]  lr=7.94e-05
  Train Loss: 0.0926                                                                                                                                                                                                                                                      
  Val Loss:   0.1536  |  Val AUC: 0.9516                                                                                                                                                                                                                                  
  No improvement for 1/7 epochs

Epoch [8/20]  lr=7.27e-05
  Train Loss: 0.0809                                                                                                                                                                                                                                                      
  Val Loss:   0.1280  |  Val AUC: 0.9610                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [9/20]  lr=6.55e-05
  Train Loss: 0.0705                                                                                                                                                                                                                                                      
  Val Loss:   0.1207  |  Val AUC: 0.9637                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [10/20]  lr=5.78e-05
  Train Loss: 0.0620                                                                                                                                                                                                                                                      
  Val Loss:   0.1280  |  Val AUC: 0.9648                                                                                                                                                                                                                                  
  No improvement for 1/7 epochs

Epoch [11/20]  lr=5.00e-05
  Train Loss: 0.0510                                                                                                                                                                                                                                                      
  Val Loss:   0.1194  |  Val AUC: 0.9648                                                                                                                                                                                                                                  
  ✓ Saved best model

Epoch [12/20]  lr=4.22e-05
  Train Loss: 0.0407                                                                                                                                                                                                                                                      
  Val Loss:   0.1250  |  Val AUC: 0.9666                                                                                                                                                                                                                                  
  No improvement for 1/7 epochs

Epoch [13/20]  lr=3.45e-05
  Train Loss: 0.0353                                                                                                                                                                                                                                                      
  Val Loss:   0.1344  |  Val AUC: 0.9642                                                                                                                                                                                                                                  
  No improvement for 2/7 epochs

Epoch [14/20]  lr=2.73e-05
  Train Loss: 0.0309                                                                                                                                                                                                                                                      
  Val Loss:   0.1434  |  Val AUC: 0.9601                                                                                                                                                                                                                                  
  No improvement for 3/7 epochs

Epoch [15/20]  lr=2.06e-05
  Train Loss: 0.0254                                                                                                                                                                                                                                                      
  Val Loss:   0.1524  |  Val AUC: 0.9629                                                                                                                                                                                                                                  
  No improvement for 4/7 epochs

Epoch [16/20]  lr=1.46e-05
  Train Loss: 0.0212                                                                                                                                                                                                                                                      
  Val Loss:   0.1373  |  Val AUC: 0.9656                                                                                                                                                                                                                                  
  No improvement for 5/7 epochs

Epoch [17/20]  lr=9.55e-06
  Train Loss: 0.0165                                                                                                                                                                                                                                                      
  Val Loss:   0.1624  |  Val AUC: 0.9599                                                                                                                                                                                                                                  
  No improvement for 6/7 epochs

Epoch [18/20]  lr=5.45e-06
  Train Loss: 0.0145                                                                                                                                                                                                                                                      
  Val Loss:   0.1494  |  Val AUC: 0.9632                                                                                                                                                                                                                                  
  No improvement for 7/7 epochs

Early stopping at epoch 18

Loaded best model (val_loss=0.1194)
                                                                                                                                                                                                                                                                          
Test Loss: 0.1170  |  Test AUC: 0.9648
                   MEL  precision=0.830  recall=0.638  f1=0.722
                    NV  precision=0.880  recall=0.938  f1=0.908
                   BCC  precision=0.895  recall=0.747  f1=0.814
                    AK  precision=0.506  recall=0.517  f1=0.511
                   BKL  precision=0.747  recall=0.653  f1=0.697
                    DF  precision=0.933  recall=0.583  f1=0.718
                  VASC  precision=0.947  recall=0.720  f1=0.818
                   SCC  precision=0.625  recall=0.476  f1=0.541
             micro avg  precision=0.844  recall=0.798  f1=0.821
             macro avg  precision=0.796  recall=0.659  f1=0.716
          weighted avg  precision=0.842  recall=0.798  f1=0.815
           samples avg  precision=0.788  recall=0.798  f1=0.792

  Auto pos_weight (log1p, max=5.0): tensor([1.7232, 0.6768, 2.0312, 3.3745, 2.2672, 4.6651, 4.6069, 3.6963])

Epoch [1/20]  lr=1.00e-04
  Train Loss: 0.4691                                                                                                                                                            
  Val Loss:   0.2721  |  Val AUC: 0.8990                                                                                                                                        
  Saved best model

Epoch [2/20]  lr=9.94e-05
  Train Loss: 0.2673                                                                                                                                                            
  Val Loss:   0.2553  |  Val AUC: 0.9287                                                                                                                                        
  Saved best model

Epoch [3/20]  lr=9.76e-05
  Train Loss: 0.2280                                                                                                                                                            
  Val Loss:   1.7547  |  Val AUC: 0.9108                                                                                                                                        
  No improvement for 1/7 epochs

Epoch [4/20]  lr=9.46e-05
  Train Loss: 0.2039                                                                                                                                                            
  Val Loss:   1.5239  |  Val AUC: 0.9119                                                                                                                                        
  No improvement for 2/7 epochs

Epoch [5/20]  lr=9.05e-05
  Train Loss: 0.1861                                                                                                                                                            
  Val Loss:   3.7739  |  Val AUC: 0.9100                                                                                                                                        
  No improvement for 3/7 epochs

Epoch [6/20]  lr=8.54e-05
  Train Loss: 0.1617                                                                                                                                                            
  Val Loss:   22.3414  |  Val AUC: 0.8842                                                                                                                                       
  No improvement for 4/7 epochs

Epoch [7/20]  lr=7.94e-05
  Train Loss: 0.1428                                                                                                                                                            
  Val Loss:   20.5246  |  Val AUC: 0.9172                                                                                                                                       
  No improvement for 5/7 epochs

Epoch [8/20]  lr=7.27e-05
  Train Loss: 0.1212                                                                                                                                                            
  Val Loss:   29.7065  |  Val AUC: 0.9199                                                                                                                                       
  No improvement for 6/7 epochs

Epoch [9/20]  lr=6.55e-05
  Train Loss: 0.1121                                                                                                                                                            
  Val Loss:   33.2691  |  Val AUC: 0.9336                                                                                                                                       
  No improvement for 7/7 epochs

Early stopping at epoch 9

Loaded best model (val_loss=0.2553)
                                                                                                                                                                                
Test Loss: 0.2077  |  Test AUC: 0.9189
                   MEL  precision=0.695  recall=0.408  f1=0.515
                    NV  precision=0.858  recall=0.825  f1=0.841
                   BCC  precision=0.682  recall=0.825  f1=0.747
                    AK  precision=0.375  recall=0.448  f1=0.408
                   BKL  precision=0.718  recall=0.302  f1=0.425
                    DF  precision=0.591  recall=0.542  f1=0.565
                  VASC  precision=0.741  recall=0.800  f1=0.769
                   SCC  precision=0.327  recall=0.587  f1=0.420
             micro avg  precision=0.749  recall=0.675  f1=0.710
             macro avg  precision=0.623  recall=0.592  f1=0.586
          weighted avg  precision=0.758  recall=0.675  f1=0.699
           samples avg  precision=0.654  recall=0.675  f1=0.661
"""
