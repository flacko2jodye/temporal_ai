
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data/raw/hmdb51"
INVENTORY_FILE = PROJECT_ROOT / "results/dataset_inventory.csv"
OUTPUT_DIR = PROJECT_ROOT / "data/processed"

CHECKPOINT_DIR = OUTPUT_DIR / "dinov2_checkpoints"
FAILURE_LOG = OUTPUT_DIR / "extraction_failures.csv"

FINAL_OUTPUT = OUTPUT_DIR / "dinov2_hmdb51_split1.npz"

NUM_FRAMES = 8
FRAME_BATCH_SIZE = 4
EMBEDDING_DIM = 384


def load_inventory():
    with INVENTORY_FILE.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def sample_video_frames(video_path, num_frames=8):
    """
    Count actual decodable frames, then sample evenly
    spaced frames using a second sequential decoding pass.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return None

    actual_frame_count = 0

    while True:
        success, frame = cap.read()

        if not success or frame is None:
            break

        actual_frame_count += 1

    cap.release()

    if actual_frame_count < num_frames:
        return None

    target_indices = np.linspace(
        0,
        actual_frame_count - 1,
        num_frames,
        dtype=int
    )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        return None

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
                cv2.COLOR_BGR2RGB
            )

            selected_frames.append(
                Image.fromarray(rgb_frame)
            )

            target_position += 1

        frame_index += 1

    cap.release()

    if len(selected_frames) != num_frames:
        return None

    return selected_frames


def extract_frame_embeddings(frames, processor, model, device):
    embeddings = []

    for start in range(0, len(frames), FRAME_BATCH_SIZE):
        batch_frames = frames[
            start:start + FRAME_BATCH_SIZE
        ]

        inputs = processor(
            images=batch_frames,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = model(**inputs)

        batch_embeddings = outputs.last_hidden_state[:, 0, :]

        embeddings.append(
            batch_embeddings.cpu().numpy()
        )

    return np.concatenate(embeddings, axis=0)


def get_checkpoint_path(category, filename):
    video_stem = Path(filename).stem

    return CHECKPOINT_DIR / category / f"{video_stem}.npz"


def is_valid_checkpoint(checkpoint_path, category, filename, split):
    if not checkpoint_path.exists():
        return False

    try:
        with np.load(checkpoint_path, allow_pickle=False) as data:
            embeddings = data["embeddings"]

            return (
                embeddings.shape == (NUM_FRAMES, EMBEDDING_DIM)
                and data["category"].item() == category
                and data["filename"].item() == filename
                and data["split"].item() == split
                and data["model_name"].item() == MODEL_NAME
            )

    except Exception:
        return False


def save_checkpoint(checkpoint_path, embeddings, category, filename, split):
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    temporary_path = checkpoint_path.with_suffix(".tmp")

    with temporary_path.open("wb") as file:
        np.savez_compressed(
            file,
            embeddings=embeddings.astype(np.float32),
            category=np.array(category),
            filename=np.array(filename),
            split=np.array(split),
            model_name=np.array(MODEL_NAME),
            num_frames=np.array(NUM_FRAMES)
        )

    temporary_path.replace(checkpoint_path)


def save_failure_log(failures):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with FAILURE_LOG.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "filename", "split", "error"]
        )

        writer.writeheader()
        writer.writerows(failures)


def consolidate_checkpoints(selected_videos):
    all_embeddings = []
    labels = []
    filenames = []
    splits = []

    for row in selected_videos:
        category = row["category"]
        filename = row["filename"]
        split = row["split"]

        checkpoint_path = get_checkpoint_path(
            category,
            filename
        )

        if not is_valid_checkpoint(
            checkpoint_path,
            category,
            filename,
            split
        ):
            continue

        with np.load(checkpoint_path, allow_pickle=False) as data:
            all_embeddings.append(data["embeddings"])
            labels.append(category)
            filenames.append(filename)
            splits.append(split)

    if not all_embeddings:
        print("No valid checkpoints available for consolidation.")
        return

    embeddings_array = np.stack(all_embeddings).astype(np.float32)

    temporary_output = FINAL_OUTPUT.with_suffix(".tmp")

    with temporary_output.open("wb") as file:
        np.savez_compressed(
            file,
            embeddings=embeddings_array,
            labels=np.array(labels),
            filenames=np.array(filenames),
            splits=np.array(splits),
            model_name=np.array(MODEL_NAME),
            num_frames=np.array(NUM_FRAMES)
        )

    temporary_output.replace(FINAL_OUTPUT)

    print("\n========== CONSOLIDATION SUMMARY ==========")
    print(f"Embedding shape: {embeddings_array.shape}")
    print(f"Videos consolidated: {len(embeddings_array)}")
    print(f"Training videos: {np.sum(np.array(splits) == 'train')}")
    print(f"Testing videos: {np.sum(np.array(splits) == 'test')}")
    print(f"Saved to: {FINAL_OUTPUT}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("Loading DINOv2 processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Loading DINOv2 model...")
    model = AutoModel.from_pretrained(MODEL_NAME)

    model.to(device)
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    inventory = load_inventory()

    selected_videos = [
        row for row in inventory
        if row["split"] in ("train", "test")
        and row["readable"].lower() == "true"
    ]

    failures = []

    total = len(selected_videos)
    successful = 0
    skipped = 0
    failed = 0

    print("\n========== EXTRACTION CONFIGURATION ==========")
    print(f"Selected videos: {total}")
    print(f"Frames per video: {NUM_FRAMES}")
    print(f"Embedding dimension: {EMBEDDING_DIM}")
    print(f"Frame batch size: {FRAME_BATCH_SIZE}")
    print(f"Checkpoint directory: {CHECKPOINT_DIR}")

    for index, row in enumerate(selected_videos, start=1):
        category = row["category"]
        filename = row["filename"]
        split = row["split"]

        video_path = DATA_DIR / category / filename

        checkpoint_path = get_checkpoint_path(
            category,
            filename
        )

        if is_valid_checkpoint(
            checkpoint_path,
            category,
            filename,
            split
        ):
            skipped += 1

        else:
            try:
                frames = sample_video_frames(
                    video_path,
                    NUM_FRAMES
                )

                if frames is None:
                    raise ValueError(
                        "Could not sample the required frames."
                    )

                video_embeddings = extract_frame_embeddings(
                    frames,
                    processor,
                    model,
                    device
                )

                if video_embeddings.shape != (
                    NUM_FRAMES,
                    EMBEDDING_DIM
                ):
                    raise ValueError(
                        f"Unexpected embedding shape: "
                        f"{video_embeddings.shape}"
                    )

                save_checkpoint(
                    checkpoint_path,
                    video_embeddings,
                    category,
                    filename,
                    split
                )

                successful += 1

            except Exception as error:
                failed += 1

                failures.append({
                    "category": category,
                    "filename": filename,
                    "split": split,
                    "error": str(error)
                })

                save_failure_log(failures)

        completed = successful + skipped + failed

        print(
            f"\rProcessed: {completed}/{total} | "
            f"New: {successful} | "
            f"Skipped: {skipped} | "
            f"Failed: {failed}",
            end="",
            flush=True
        )

    print("\n\n========== EXTRACTION SUMMARY ==========")
    print(f"Selected videos: {total}")
    print(f"Newly extracted: {successful}")
    print(f"Skipped (existing checkpoints): {skipped}")
    print(f"Failed: {failed}")

    save_failure_log(failures)

    consolidate_checkpoints(selected_videos)

    if failures:
        print(f"\nFailure log: {FAILURE_LOG}")


if __name__ == "__main__":
    main()
