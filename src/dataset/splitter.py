from pathlib import Path
import zipfile
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_COLS = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC", "UNK"]

def extract_archive_if_needed():
    dir = Path("src/dataset")
    archive_dir = dir / "archive"
    zip_file = dir / "archive.zip"

    if archive_dir.is_dir():
        if any(archive_dir.iterdir()): 
            if zip_file.is_file():
                zip_file.unlink()
                print("The 'archive' subdirectory already exists. 'archive.zip' deleted.")
            else:
                print("The 'archive' subdirectory already exists.")
            return
        else:
            print("The 'archive' directory exists but is empty. Proceeding with extraction...")
    
    if zip_file.is_file():
        try:
            print("Extracting 'archive.zip'")
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(dir)
            print("Extraction complete.")
            
            zip_file.unlink()
            print("'archive.zip' has been deleted.")
            
        except (zipfile.BadZipFile, PermissionError) as e:
            print(f"Extraction failed: {e}")
            print("'archive.zip' was not deleted.")
    else:
        print("Neither the 'archive' folder nor 'archive.zip' was found.")


def drop_zero_variance_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Drops only numeric columns with a mathematical variance of exactly 0."""

    numeric_df = df.select_dtypes(include='number')
    
    zero_var_cols = numeric_df.columns[numeric_df.var() == 0]
    
    return df.drop(columns=zero_var_cols)


def load_ground_truth(gt_path: Path) -> pd.DataFrame:
    df = pd.read_csv(gt_path)
    df["label"] = df[LABEL_COLS].idxmax(axis=1)
    df = drop_zero_variance_numeric(df)
    return df


def stratified_split(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6

    rare_mask = df["label"].isin(
        df["label"].value_counts()[lambda x: x < 4].index
    )
    rare_df = df[rare_mask]
    main_df = df[~rare_mask]

    train_df, temp_df = train_test_split(
        main_df, test_size=(1 - train_size),
        stratify=main_df["label"], random_state=seed,
    )
    val_ratio_adjusted = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_ratio_adjusted),
        stratify=temp_df["label"], random_state=seed,
    )

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

    return train_df.drop(columns=['label']), val_df.drop(columns=['label']), test_df.drop(columns=['label'])


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
    extract_archive_if_needed()

    gt_path = Path(
        "src/dataset/archive/ISIC_2019_Training_GroundTruth.csv")
    split_dir = Path("src/dataset/after_split")

    df = load_ground_truth(gt_path)
    train_df, val_df, test_df = stratified_split(df)

    save_splits(train_df, val_df, test_df, split_dir)


if __name__ == "__main__":
    main()
