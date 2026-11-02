
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INVENTORY = Path("results/dataset_inventory.csv")
DATA_ROOT = Path("data/raw/hmdb51")
OUTPUT_DIR = Path("results/visualizations")

CATEGORIES = [
    "brush_hair",
    "clap",
    "drink",
    "eat",
    "pour",
    "wave",
]

FRAMES_PER_VIDEO = 8
RANDOM_SEED = 42

TRAIN_SAMPLES_PER_CLASS = 2
TEST_SAMPLES_PER_CLASS = 1


# --------------------------------------------------
# Frame sampling
# --------------------------------------------------

def sample_video_frames(video_path, num_frames=8):
    """
    Sample uniformly spaced frames using sequential decoding.

    Pass 1:
        Count successfully decoded frames.

    Pass 2:
        Select evenly spaced frames using np.linspace.

    This follows the sampling strategy used by the
    DINOv2 embedding extraction pipeline.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return []

    actual_frame_count = 0

    # First pass: count decodable frames
    while True:
        success, frame = cap.read()

        if not success or frame is None:
            break

        actual_frame_count += 1

    cap.release()

    if actual_frame_count < num_frames:
        return []

    target_indices = np.linspace(
        0,
        actual_frame_count - 1,
        num_frames,
        dtype=int,
    )

    # Second pass: retrieve selected frames
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return []

    selected_frames = []
    frame_index = 0
    target_position = 0

    while target_position < num_frames:
        success, frame = cap.read()

        if not success or frame is None:
            break

        if frame_index == target_indices[target_position]:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            selected_frames.append(rgb_frame)
            target_position += 1

        frame_index += 1

    cap.release()

    if len(selected_frames) != num_frames:
        return []

    return selected_frames


# --------------------------------------------------
# Select representative videos
# --------------------------------------------------

def choose_videos(category_df, rng):
    train_rows = category_df[
        category_df["split"] == "train"
    ]

    test_rows = category_df[
        category_df["split"] == "test"
    ]

    train_records = train_rows.to_dict("records")
    test_records = test_rows.to_dict("records")

    selected = []

    if train_records:
        selected.extend(
            rng.sample(
                train_records,
                min(
                    TRAIN_SAMPLES_PER_CLASS,
                    len(train_records),
                ),
            )
        )

    if test_records:
        selected.extend(
            rng.sample(
                test_records,
                min(
                    TEST_SAMPLES_PER_CLASS,
                    len(test_records),
                ),
            )
        )

    return selected


# --------------------------------------------------
# Generate temporal frame strips
# --------------------------------------------------

def render_category_strip(category, selected_rows):
    if not selected_rows:
        print(
            f"No readable videos found for category: {category}"
        )
        return

    num_rows = len(selected_rows)

    fig, axes = plt.subplots(
        num_rows,
        FRAMES_PER_VIDEO,
        figsize=(
            2.2 * FRAMES_PER_VIDEO,
            2.4 * num_rows,
        ),
        squeeze=False,
    )

    fig.suptitle(
        f"Temporal Frame Sequences: {category}",
        fontsize=16,
        y=0.98,
    )

    for row_idx, row in enumerate(selected_rows):
        filename = row["filename"]
        split = row["split"]

        video_path = (
            DATA_ROOT / category / filename
        )

        frames = sample_video_frames(
            video_path,
            FRAMES_PER_VIDEO,
        )

        if len(frames) != FRAMES_PER_VIDEO:
            print(
                f"Warning: Could not sample "
                f"{FRAMES_PER_VIDEO} frames from {video_path}"
            )

        for col_idx in range(FRAMES_PER_VIDEO):
            ax = axes[row_idx, col_idx]

            if col_idx < len(frames):
                ax.imshow(frames[col_idx])
            else:
                ax.text(
                    0.5,
                    0.5,
                    "Missing",
                    ha="center",
                    va="center",
                    fontsize=8,
                    transform=ax.transAxes,
                )

            ax.axis("off")

            if row_idx == 0:
                ax.set_title(
                    f"Frame {col_idx + 1}",
                    fontsize=10,
                )

        row_label = f"{split}: {filename}"

        axes[row_idx, 0].set_ylabel(
            row_label[:40],
            fontsize=8,
            rotation=0,
            labelpad=55,
            va="center",
        )

    plt.tight_layout(
        rect=(0.03, 0.02, 1, 0.95)
    )

    output_path = (
        OUTPUT_DIR /
        f"{category}_temporal_strip.png"
    )

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not INVENTORY.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INVENTORY}"
        )

    df = pd.read_csv(INVENTORY)

    required_columns = {
        "category",
        "filename",
        "split",
        "readable",
    }

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            "Inventory is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    readable_mask = (
        df["readable"]
        .astype(str)
        .str.lower()
        == "true"
    )

    df = df[readable_mask].copy()

    rng = random.Random(RANDOM_SEED)

    for category in CATEGORIES:
        category_df = df[
            df["category"] == category
        ].copy()

        selected_rows = choose_videos(
            category_df,
            rng,
        )

        render_category_strip(
            category,
            selected_rows,
        )


if __name__ == "__main__":
    main()
