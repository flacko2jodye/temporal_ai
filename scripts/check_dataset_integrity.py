from pathlib import Path

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


def read_split_file(path):
    entries = {}

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 2:
                continue

            filename, split_id = parts
            entries[filename] = int(split_id)

    return entries


def main():
    total_missing = 0

    for category in CATEGORIES:
        video_dir = DATA_DIR / category
        split_file = SPLIT_DIR / f"{category}_test_split1.txt"

        available = {
            file.name for file in video_dir.glob("*.avi")
        }

        split_entries = read_split_file(split_file)

        expected = set(split_entries)
        missing = expected - available
        extra = available - expected

        total_missing += len(missing)

        print(f"\nCategory: {category}")
        print(f"Available videos: {len(available)}")
        print(f"Split entries: {len(expected)}")
        print(f"Missing videos: {len(missing)}")
        print(f"Extra videos: {len(extra)}")

        for filename in sorted(missing):
            print(f"  MISSING: {filename}")

    print(f"\nTotal missing videos: {total_missing}")


if __name__ == "__main__":
    main()