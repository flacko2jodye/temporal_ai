import numpy as np
from pathlib import Path
from collections import Counter


EMBEDDING_FILE = Path(
    "data/processed/dinov2_hmdb51_split1.npz"
)

EXPECTED_MODEL = "facebook/dinov2-small"
EXPECTED_SHAPE = (600, 8, 384)
EXPECTED_TRAIN = 420
EXPECTED_TEST = 180

CATEGORIES = [
    "brush_hair",
    "clap",
    "drink",
    "eat",
    "pour",
    "wave",
]


def main():
    print("Loading embedding dataset...")

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(EMBEDDING_FILE)

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
        filenames = data["filenames"]
        splits = data["splits"]
        model_name = data["model_name"].item()
        num_frames = int(data["num_frames"].item())

    print("\n========== BASIC VALIDATION ==========")

    print("Embedding shape:", embeddings.shape)
    print("Data type:", embeddings.dtype)
    print("Model:", model_name)
    print("Frames per video:", num_frames)
    print("Total videos:", len(embeddings))

    assert embeddings.shape == EXPECTED_SHAPE, (
        f"Unexpected embedding shape: {embeddings.shape}"
    )

    assert model_name == EXPECTED_MODEL
    assert num_frames == 8
    assert embeddings.dtype == np.float32

    print("\n========== DATA INTEGRITY ==========")

    print("NaN values:", np.isnan(embeddings).sum())
    print("Infinite values:", np.isinf(embeddings).sum())

    assert np.isfinite(embeddings).all(), (
        "Embeddings contain NaN or infinite values."
    )

    assert len(labels) == len(embeddings)
    assert len(filenames) == len(embeddings)
    assert len(splits) == len(embeddings)

    assert len(set(filenames.tolist())) == len(filenames), (
        "Duplicate filenames detected."
    )

    print("All embeddings are finite.")
    print("No duplicate filenames detected.")

    print("\n========== SPLIT DISTRIBUTION ==========")

    split_counts = Counter(splits.tolist())

    for split, count in sorted(split_counts.items()):
        print(f"{split}: {count}")

    assert split_counts["train"] == EXPECTED_TRAIN
    assert split_counts["test"] == EXPECTED_TEST

    print("\n========== CATEGORY DISTRIBUTION ==========")

    label_counts = Counter(labels.tolist())

    for category in CATEGORIES:
        print(f"{category}: {label_counts[category]}")

    assert set(label_counts.keys()) == set(CATEGORIES)

    print("\n========== CATEGORY / SPLIT DISTRIBUTION ==========")

    for category in CATEGORIES:
        train_count = sum(
            (labels == category) & (splits == "train")
        )

        test_count = sum(
            (labels == category) & (splits == "test")
        )

        print(
            f"{category:12} | "
            f"Train: {train_count:3} | "
            f"Test: {test_count:3} | "
            f"Total: {train_count + test_count:3}"
        )

        assert train_count == 70
        assert test_count == 30

    print("\n========== EMBEDDING STATISTICS ==========")

    print("Minimum:", float(embeddings.min()))
    print("Maximum:", float(embeddings.max()))
    print("Mean:", float(embeddings.mean()))
    print("Standard deviation:", float(embeddings.std()))

    print("\n========== VALIDATION RESULT ==========")
    print("All validation checks passed!")


if __name__ == "__main__":
    main()