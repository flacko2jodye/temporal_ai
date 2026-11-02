import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.extract_embeddings import sample_video_frames


DATA_DIR = PROJECT_ROOT / "data/raw/hmdb51"
INVENTORY_FILE = PROJECT_ROOT / "results/dataset_inventory.csv"

NUM_FRAMES = 8
TEST_LIMIT = None  # None means test all selected videos


def main():
    with INVENTORY_FILE.open("r", encoding="utf-8") as file:
        inventory = list(csv.DictReader(file))

    selected = [
        row for row in inventory
        if row["split"] in ("train", "test")
        and row["readable"].lower() == "true"
    ]

    videos_to_test = (
        selected if TEST_LIMIT is None
        else selected[:TEST_LIMIT]
    )

    print(f"Selected videos: {len(selected)}")
    print(f"Testing {len(videos_to_test)} videos\n")

    successful = 0
    failed = 0
    failed_videos = []

    for index, row in enumerate(videos_to_test, start=1):
        category = row["category"]
        filename = row["filename"]
        video_path = DATA_DIR / category / filename

        try:
            frames = sample_video_frames(
                video_path,
                NUM_FRAMES
            )

            if frames is None:
                failed += 1
                failed_videos.append(filename)
                status = "FAILED"

            else:
                successful += 1
                status = f"OK ({len(frames)} frames)"

        except Exception as error:
            failed += 1
            failed_videos.append(filename)
            status = f"ERROR: {error}"

        print(
            f"\rProcessed {index}/{len(videos_to_test)} | "
            f"Successful: {successful} | Failed: {failed}",
            end="",
            flush=True
        )

        if status.startswith("FAILED") or status.startswith("ERROR"):
            print(f"\n{filename}: {status}")

    print("\n\n========== DIAGNOSTIC SUMMARY ==========")
    print(f"Selected videos: {len(selected)}")
    print(f"Tested videos: {len(videos_to_test)}")
    print(f"Sampling successful: {successful}")
    print(f"Sampling failed: {failed}")

    if failed_videos:
        print("\nFailed videos:")

        for filename in failed_videos:
            print(filename)


if __name__ == "__main__":
    main()