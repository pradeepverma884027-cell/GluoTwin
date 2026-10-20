"""Run this first. Inspects the prepared dataset without changing it."""
from common import DATASET_NAME, TARGET_COLUMN, load_data


def main():
    df, features, targets, schema = load_data(DATASET_NAME)
    print("=" * 65)
    print("DATASET INSPECTION")
    print("Dataset:", DATASET_NAME)
    print("Rows, columns:", df.shape)
    print("Features in schema:", len(features))
    print("Targets:", targets)
    print("\nColumns:")
    print(df.columns.tolist())
    print("\nData splits:")
    print(df["split"].value_counts(dropna=False))
    print("\nMissing values in feature columns:")
    missing = df[features].isna().sum()
    print(missing[missing > 0].sort_values(ascending=False).to_string() or "None")
    print("\nDuplicate rows:", int(df.duplicated().sum()))
    if TARGET_COLUMN in df.columns:
        print(f"\nTarget summary: {TARGET_COLUMN}")
        print(df[TARGET_COLUMN].describe())
        print("Unique target values:", df[TARGET_COLUMN].nunique())
    print("\nNo files were modified.")

if __name__ == "__main__":
    main()
