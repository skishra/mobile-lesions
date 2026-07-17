import torch
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report
from tqdm import tqdm
import torch.nn as nn


class Evaluator:
    def __init__(self, model, test_loader, device=None, class_names=None):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss()

    @torch.no_grad()
    def evaluate(self):
        all_probs = []
        all_labels = []
        total_loss = 0.0

        loop = tqdm(self.test_loader, desc="Evaluating", leave=False)

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

        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        avg_loss = total_loss / len(self.test_loader)

        num_classes = len(
            self.class_names) if self.class_names else all_probs.shape[1]
        per_class_auc = {}

        for i in range(num_classes):
            binary_labels_for_class = (all_labels == i).astype(int)

            if len(np.unique(binary_labels_for_class)) < 2:
                per_class_auc[i] = float("nan")
            else:
                per_class_auc[i] = roc_auc_score(
                    binary_labels_for_class, all_probs[:, i]
                )

        try:
            macro_auc = roc_auc_score(
                all_labels, all_probs, average="macro", multi_class="ovr"
            )
        except ValueError:
            macro_auc = 0.0

        preds = np.argmax(all_probs, axis=1)
        report = classification_report(
            all_labels, preds,
            target_names=self.class_names,
            zero_division=0,
            output_dict=True,
        )

        return {
            "loss": avg_loss,
            "macro_auc": macro_auc,
            "per_class_auc": per_class_auc,
            "report": report,
        }


def main():
    from pathlib import Path
    import timm
    from torch.utils.data import DataLoader
    from torchvision import transforms
    from src.dataset.dataset import ISIC2019Dataset

    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    test_csv = Path("src/dataset/archive/after_split/test.csv")
    image_dir = Path("src/dataset/lesions-kaggle/ISIC_2019_Training_Input/")

    test_dataset = ISIC2019Dataset(
        csv_file=test_csv,
        image_dir=image_dir,
        transform=transform,
    )

    test_loader = DataLoader(
        test_dataset, batch_size=64, num_workers=4, pin_memory=True,
    )

    model = timm.create_model(
        "mobilenetv4_conv_medium.e500_r256_in1k",
        pretrained=False,
        num_classes=len(test_dataset.classes),
    )
    model.load_state_dict(torch.load("best_model.pt", map_location="cpu"))

    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=test_dataset.classes,
    )
    results = evaluator.evaluate()

    print(f"\nTest Loss:     {results['loss']:.4f}")
    print(f"Macro ROC AUC: {results['macro_auc']:.4f}")

    print("\nPer-class ROC AUC:")
    for cls_idx, auc in results["per_class_auc"].items():
        name = test_dataset.classes[cls_idx]
        print(f"  {name}: {auc:.4f}" if not np.isnan(auc)
              else f"  {name}: N/A (single class in test set)")

    print("\nClassification Report:")
    for cls, metrics in results["report"].items():
        if isinstance(metrics, dict):
            print(f"  {cls:>20s}  precision={metrics['precision']:.3f}  "
                  f"recall={metrics['recall']:.3f}  f1={metrics['f1-score']:.3f}")


if __name__ == "__main__":
    main()
