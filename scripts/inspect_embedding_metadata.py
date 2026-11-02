from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dinov2_hmdb51_split1.npz"
)


def main():
    if not EMBEDDING_FILE.exists():
        print(f"Embedding file not found: {EMBEDDING_FILE}")
        return

    with np.load(EMBEDDING_FILE, allow_pickle=True) as data:
        print("=" * 70)
        print("EMBEDDING FILE METADATA")
        print("=" * 70)
        print("File:", EMBEDDING_FILE)
        print("\nAvailable keys:")

        for key in data.files:
            array = data[key]

            print(f"\nKey: {key}")
            print("  Shape:", array.shape)
            print("  Data type:", array.dtype)

            if array.size == 0:
                print("  Empty array")
            elif array.ndim == 0:
                print("  Value:", array.item())
            elif array.dtype.kind in ("U", "S") or array.dtype == object:
                print("  First 5 values:", array[:5])
            elif array.ndim == 1:
                print("  First 5 values:", array[:5])
            else:
                print("  First value:", array[0])


if __name__ == "__main__":
    main()