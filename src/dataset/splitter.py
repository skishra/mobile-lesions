from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    df["label"] = df[LABEL_COLS].idxmax(axis=1)
    return df


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    # For classes with very few samples, force at least 1 into each split
    rare_mask = df["label"].isin(
        df["label"].value_counts()[lambda x: x < 4].index
    )
    rare_df = df[rare_mask]
    main_df = df[~rare_mask]

    # Split the main (non-rare) portion normally
    train_df, temp_df = train_test_split(
        main_df, test_size=(1 - train_size),
        stratify=main_df["label"], random_state=seed,
    )
    val_ratio_adjusted = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_adjusted),
        stratify=temp_df["label"], random_state=seed,
    )

    # Distribute rare samples round-robin across splits
    for _, group in rare_df.groupby("label"):
        rows = group.sample(frac=1, random_state=seed).reset_index(drop=True)
        splits = [train_df, val_df, test_df]
        for i, (_, row) in enumerate(rows.iterrows()):
            splits[i % 3] = pd.concat([splits[i % 3], row.to_frame().T])
        train_df, val_df, test_df = splits

    def check_dist(df, name):
        print(f"\n{name}")
        print(df["label"].value_counts(normalize=True))

    check_dist(train_df, "Train")
    check_dist(val_df, "Val")
    check_dist(test_df, "Test")

    return train_df.drop(columns=['label', 'UNK']), val_df.drop(columns=['label', 'UNK']), test_df.drop(columns=['label', 'UNK'])


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(out_dir / "train.csv", index=False)
    val_df.to_csv(out_dir / "val.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)


def main():
    gt_path = Path(
        "src/dataset/lesions-kaggle/ISIC_2019_Training_GroundTruth.csv")
    split_dir = Path("src/dataset/after_split")

    df = load_ground_truth(gt_path)
    train_df, val_df, test_df = stratified_split(df)

    save_splits(train_df, val_df, test_df, split_dir)


if __name__ == "__main__":
    main()
