import csv
from pathlib import Path

import cv2


CATEGORIES = [
    "brush_hair",
    "clap",
    "drink",
    "eat",
    "pour",
    "wave",
]

DATA_DIR = Path("data/raw/hmdb51")
SPLIT_DIR = Path("data/splits/testTrainMulti_7030_splits")
OUTPUT_FILE = Path("results/dataset_inventory.csv")


def load_split_assignments(category):
    split_file = SPLIT_DIR / f"{category}_test_split1.txt"
    assignments = {}

    with split_file.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 2:
                continue

            filename, split_id = parts

            split_names = {
                "0": "excluded",
                "1": "train",
                "2": "test",
            }

            assignments[filename] = split_names[split_id]

    return assignments


def inspect_video(video_path):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return {
            "readable": False,
            "frame_count": 0,
            "fps": 0,
            "width": 0,
            "height": 0,
            "duration_seconds": 0,
        }

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    success, frame = cap.read()
    first_frame_read = success and frame is not None

    duration = frame_count / fps if fps > 0 else 0

    cap.release()

    return {
        "readable": first_frame_read,
        "frame_count": frame_count,
        "fps": round(fps, 2),
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 2),
    }


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    records = []

    for category in CATEGORIES:
        print(f"\nInspecting category: {category}")

        assignments = load_split_assignments(category)
        video_files = sorted(
            (DATA_DIR / category).glob("*.avi")
        )

        for index, video_path in enumerate(video_files, start=1):
            metadata = inspect_video(video_path)

            records.append({
                "category": category,
                "filename": video_path.name,
                "split": assignments.get(video_path.name, "unknown"),
                **metadata,
            })

            print(
                f"\rProcessed {index}/{len(video_files)}",
                end="",
                flush=True,
            )

        print()

    fieldnames = [
        "category",
        "filename",
        "split",
        "readable",
        "frame_count",
        "fps",
        "width",
        "height",
        "duration_seconds",
    ]

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    total = len(records)
    readable = sum(row["readable"] for row in records)
    unreadable = total - readable

    print("\n========== DATASET INSPECTION SUMMARY ==========")
    print(f"Total videos: {total}")
    print(f"Readable videos: {readable}")
    print(f"Unreadable videos: {unreadable}")

    print("\nSplit distribution:")

    for split in ["train", "test", "excluded", "unknown"]:
        count = sum(row["split"] == split for row in records)
        print(f"{split}: {count}")

    print(f"\nInventory saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()