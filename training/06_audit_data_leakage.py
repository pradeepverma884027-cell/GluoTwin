
from common import load_data
import pandas as pd

DATASET_NAME = "cgm_t1dm_t2dm"
TARGET_COLUMN = "target_cgm_t30"


def main():
    df, features, targets, schema = load_data(DATASET_NAME)

    print("=" * 65)
    print("GLUCOTWIN DATA LEAKAGE AUDIT")
    print("=" * 65)

    # 1. Check whether identifiers or target columns are features.
    forbidden = {
        "patient_uid", "segment_id", "dt", "split"
    }

    print("\n1. IDENTIFIER / METADATA CHECK")
    print("Identifier columns used as features:",
          sorted(set(features) & forbidden))

    print("Target columns included as features:",
          sorted(set(features) & set(targets)))

    # 2. Check patient overlap between splits.
    if "patient_uid" in df.columns and "split" in df.columns:
        print("\n2. PATIENT OVERLAP")

        patient_sets = {}

        for split_name in ["train", "val", "test"]:
            part = df[
                df["split"].astype(str).str.lower()
                == split_name
            ]

            patient_sets[split_name] = set(
                part["patient_uid"].dropna().astype(str)
            )

            print(
                f"{split_name}: {len(patient_sets[split_name])} "
                "unique patients"
            )

        for a, b in [
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ]:
            overlap = patient_sets[a] & patient_sets[b]
            print(f"{a} / {b} overlap: {len(overlap)} patients")

    # 3. Check segment overlap, using patient + segment identity.
    if {"patient_uid", "segment_id", "split"}.issubset(df.columns):
        print("\n3. SEGMENT OVERLAP")

        segment_sets = {}

        for split_name in ["train", "val", "test"]:
            part = df[
                df["split"].astype(str).str.lower()
                == split_name
            ]

            segment_sets[split_name] = set(
                zip(
                    part["patient_uid"].astype(str),
                    part["segment_id"].astype(str),
                )
            )

        for a, b in [
            ("train", "val"),
            ("train", "test"),
            ("val", "test"),
        ]:
            overlap = segment_sets[a] & segment_sets[b]
            print(f"{a} / {b} overlap: {len(overlap)} segments")

    # 4. Inspect time ranges in each split.
    if "dt" in df.columns:
        print("\n4. TIME RANGES")

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")

        for split_name in ["train", "val", "test"]:
            part = df[
                df["split"].astype(str).str.lower()
                == split_name
            ]

            print(
                f"{split_name}: "
                f"{part['dt'].min()} -> {part['dt'].max()}"
            )

    # 5. Inspect target statistics in each split.
    print("\n5. TARGET DISTRIBUTION")

    for split_name in ["train", "val", "test"]:
        part = df[
            df["split"].astype(str).str.lower()
            == split_name
        ]

        y = pd.to_numeric(
            part[TARGET_COLUMN], errors="coerce"
        ).dropna()

        print(
            f"{split_name}: n={len(y):,}, "
            f"mean={y.mean():.2f}, "
            f"std={y.std():.2f}, "
            f"min={y.min():.2f}, "
            f"max={y.max():.2f}"
        )

    print("\n6. SCHEMA FEATURES")
    print(features)

    print("\nAudit complete. No files were modified.")
    print("Review overlaps and feature engineering before conclusions.")


if __name__ == "__main__":
    main()
