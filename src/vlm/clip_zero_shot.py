
import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import CLIPModel, CLIPProcessor


MODEL_NAME = "openai/clip-vit-base-patch32"
NUM_FRAMES = 8

# Human-readable prompts for zero-shot classification.
CLASS_PROMPTS = {
    "brush_hair": "A video of a person brushing their hair.",
    "clap": "A video of a person clapping their hands.",
    "drink": "A video of a person drinking something.",
    "eat": "A video of a person eating food.",
    "pour": "A video of a person pouring a liquid.",
    "wave": "A video of a person waving their hand.",
}

CLASS_NAMES = list(CLASS_PROMPTS.keys())


def load_video_frames(video_path, num_frames=NUM_FRAMES):
    """Sample evenly spaced RGB frames from a video."""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        raise RuntimeError(f"Invalid frame count: {video_path}")

    indices = np.linspace(
        0, total_frames - 1, num_frames, dtype=int
    )

    frames = []

    for frame_index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame))

    cap.release()

    if not frames:
        raise RuntimeError(f"No readable frames: {video_path}")

    return frames


def build_video_index(data_root):
    """Index AVI files once instead of searching for each video."""
    video_index = {}

    for path in Path(data_root).rglob("*.avi"):
        video_index.setdefault(path.name, []).append(path)

    return video_index


def load_clip(device):
    print(f"Loading CLIP model: {MODEL_NAME}")

    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model = CLIPModel.from_pretrained(MODEL_NAME)
    model = model.to(device)
    model.eval()

    return model, processor


def encode_text_prompts(model, processor, device):
    prompts = [CLASS_PROMPTS[name] for name in CLASS_NAMES]

    inputs = processor(
        text=prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)

        if not isinstance(text_features, torch.Tensor):
            text_features = text_features.pooler_output

        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )

    return text_features


def classify_video(
    video_path,
    model,
    processor,
    text_features,
    device,
    num_frames=NUM_FRAMES,
):
    """
    Score every sampled frame against all class prompts.
    Average frame-level similarity scores for one video prediction.
    """
    frames = load_video_frames(video_path, num_frames)

    inputs = processor(
        images=frames,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        image_features = model.get_image_features(
            pixel_values=pixel_values
        )

        if not isinstance(image_features, torch.Tensor):
            image_features = image_features.pooler_output

        image_features = image_features / image_features.norm(
            dim=-1, keepdim=True
        )

        # Shape: sampled_frames x classes
        frame_scores = image_features @ text_features.T

        # Aggregate frame scores into one video-level score.
        video_scores = frame_scores.mean(dim=0)

        probabilities = torch.softmax(video_scores, dim=0)

    predicted_index = int(torch.argmax(video_scores).item())

    return {
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "scores": video_scores.cpu().numpy(),
        "num_frames_used": len(frames),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--inventory",
        type=str,
        default="results/dataset_inventory.csv",
    )

    parser.add_argument(
        "--data-root",
        type=str,
        default="data",
    )

    parser.add_argument(
        "--num-frames",
        type=int,
        default=NUM_FRAMES,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    inventory_path = Path(args.inventory)

    if not inventory_path.exists():
        raise FileNotFoundError(inventory_path)

    inventory = pd.read_csv(inventory_path)

    # Strictly evaluate the official test split.
    test_data = inventory[
        (inventory["split"] == "test")
        & (inventory["readable"] == True)
    ].copy()

    print(f"Test videos found in inventory: {len(test_data)}")

    video_index = build_video_index(args.data_root)

    model, processor = load_clip(device)

    text_features = encode_text_prompts(
        model, processor, device
    )

    results = []
    failures = []

    for index, row in enumerate(
        test_data.itertuples(index=False), start=1
    ):
        filename = row.filename
        category = row.category

        matches = video_index.get(filename, [])

        if len(matches) == 0:
            failures.append({
                "filename": filename,
                "error": "Video file not found",
            })
            print(f"[{index}/{len(test_data)}] MISSING: {filename}")
            continue

        if len(matches) > 1:
            failures.append({
                "filename": filename,
                "error": "Duplicate filename found",
            })
            print(f"[{index}/{len(test_data)}] DUPLICATE: {filename}")
            continue

        try:
            prediction = classify_video(
                matches[0],
                model,
                processor,
                text_features,
                device,
                num_frames=args.num_frames,
            )

            result = {
                "filename": filename,
                "true_class": category,
                "predicted_class": prediction["predicted_class"],
                "confidence": prediction["confidence"],
                "num_frames_used": prediction["num_frames_used"],
            }

            # Keep each class score for later analysis.
            for class_name, score in zip(
                CLASS_NAMES, prediction["scores"]
            ):
                result[f"score_{class_name}"] = float(score)

            results.append(result)

            print(
                f"[{index}/{len(test_data)}] "
                f"{filename}: "
                f"true={category}, "
                f"pred={prediction['predicted_class']}"
            )

        except Exception as exc:
            failures.append({
                "filename": filename,
                "error": str(exc),
            })
            print(f"[{index}/{len(test_data)}] ERROR: {filename}: {exc}")

    if not results:
        raise RuntimeError("No videos were successfully evaluated.")

    output_dir = Path("results/clip_zero_shot")
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(results)

    y_true = results_df["true_class"]
    y_pred = results_df["predicted_class"]

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model": MODEL_NAME,
        "experiment": "zero_shot_clip_multi_frame_score_averaging",
        "num_frames": args.num_frames,
        "num_test_videos_in_inventory": len(test_data),
        "num_evaluated": len(results),
        "num_failures": len(failures),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "per_class_metrics": report,
        "class_prompts": CLASS_PROMPTS,
        "device": str(device),
    }

    results_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    with open(
        output_dir / "metrics.json", "w", encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    with open(
        output_dir / "failures.json", "w", encoding="utf-8"
    ) as file:
        json.dump(failures, file, indent=4)

    # Confusion matrix.
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_NAMES,
    )

    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(image, ax=ax)

    ax.set(
        xticks=np.arange(len(CLASS_NAMES)),
        yticks=np.arange(len(CLASS_NAMES)),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted class",
        ylabel="True class",
        title="Zero-Shot CLIP Confusion Matrix",
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=35,
        ha="right",
        rotation_mode="anchor",
    )

    threshold = cm.max() / 2 if cm.size else 0

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(
        plots_dir / "confusion_matrix.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    print("\n========== CLIP ZERO-SHOT RESULTS ==========")
    print(f"Model: {MODEL_NAME}")
    print(f"Videos evaluated: {len(results)}/{len(test_data)}")
    print(f"Failures: {len(failures)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Results saved to: {output_dir.resolve()}")
    print("============================================")


if __name__ == "__main__":
    main()
