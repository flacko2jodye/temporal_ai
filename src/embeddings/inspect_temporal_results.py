
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ANALYSIS_DIR = PROJECT_ROOT / "results" / "temporal_analysis"

CSV_FILES = {
    "CLASS TEMPORAL STATISTICS": "class_temporal_statistics.csv",
    "CLASS CENTROID SIMILARITY": "class_centroid_similarity.csv",
    "PER-VIDEO TEMPORAL STATISTICS": "per_video_temporal_statistics.csv",
}


def inspect_class_statistics():

    path = ANALYSIS_DIR / CSV_FILES["CLASS TEMPORAL STATISTICS"]
    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print("1. TEMPORAL VARIATION BY ACTION CLASS")
    print("=" * 70)

    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nClasses ranked by temporal variation:")

    ranked = df.sort_values(
        "mean_temporal_variation",
        ascending=False,
    )

    print(
        ranked[
            ["class", "mean_temporal_variation", "std_temporal_variation"]
        ].to_string(index=False)
    )


def inspect_centroid_similarity():

    path = ANALYSIS_DIR / CSV_FILES["CLASS CENTROID SIMILARITY"]
    df = pd.read_csv(path, index_col=0)

    print("\n" + "=" * 70)
    print("2. CLASS CENTROID COSINE SIMILARITY")
    print("=" * 70)

    print(df.round(4).to_string())

    values = []

    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            values.append(
                (df.index[i], df.columns[j], df.iloc[i, j])
            )

    values.sort(key=lambda item: item[2], reverse=True)

    print("\nMost similar class pairs:")

    for class_a, class_b, score in values[:5]:
        print(f"{class_a:12s} <-> {class_b:12s}: {score:.4f}")

    print("\nLeast similar class pairs:")

    for class_a, class_b, score in values[-5:]:
        print(f"{class_a:12s} <-> {class_b:12s}: {score:.4f}")


def inspect_video_statistics():

    path = ANALYSIS_DIR / CSV_FILES["PER-VIDEO TEMPORAL STATISTICS"]
    df = pd.read_csv(path)

    print("\n" + "=" * 70)
    print("3. PER-VIDEO TEMPORAL STATISTICS")
    print("=" * 70)

    print(f"Total videos: {len(df)}")

    print("\nAverage temporal variation by class:")

    print(
        df.groupby("label")["mean_temporal_variation"]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )

    print("\nHighest temporal variation videos:")

    print(
        df.nlargest(10, "mean_temporal_variation")[
            ["filename", "label", "split", "mean_temporal_variation"]
        ].to_string(index=False)
    )

    print("\nLowest temporal variation videos:")

    print(
        df.nsmallest(10, "mean_temporal_variation")[
            ["filename", "label", "split", "mean_temporal_variation"]
        ].to_string(index=False)
    )


def main():

    print("TEMPORAL ANALYSIS INSPECTION")
    print("Reading previously generated CSV files...")

    for name, filename in CSV_FILES.items():

        path = ANALYSIS_DIR / filename

        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    inspect_class_statistics()
    inspect_centroid_similarity()
    inspect_video_statistics()

    print("\n" + "=" * 70)
    print("INSPECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
