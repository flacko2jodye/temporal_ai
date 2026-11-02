
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageDraw
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import AutoImageProcessor, AutoModel
from src.embeddings.extract_embeddings import (
    DATA_DIR,
    MODEL_NAME,
    NUM_FRAMES,
    load_inventory,
    sample_video_frames,
    extract_frame_embeddings,
)

from src.models.checkpoint_utils import load_temporal_gru
from src.evaluation.common import (
    load_test_inventory,
    evaluate_gru_predictions,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_FILE = PROJECT_ROOT / "results/gru_temporal.pth"

RESULTS_DIR = PROJECT_ROOT / "results/image_occlusion"
PLOT_DIR = RESULTS_DIR / "plots"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

EXPECTED_TEST_VIDEOS = 180

# Occlusion covers 25% of image width and height.
OCCLUSION_WIDTH_RATIO = 0.25
OCCLUSION_HEIGHT_RATIO = 0.25

OCCLUSION_COLOR = (0, 0, 0)

# ============================================================
# IMAGE-LEVEL OCCLUSION
# Apply a black square to the centre of each RGB frame.
# ============================================================

def apply_central_occlusion(image):

    width, height = image.size

    patch_width = max(
        1,
        int(width * OCCLUSION_WIDTH_RATIO),
    )

    patch_height = max(
        1,
        int(height * OCCLUSION_HEIGHT_RATIO),
    )

    left = (width - patch_width) // 2
    top = (height - patch_height) // 2

    right = left + patch_width
    bottom = top + patch_height

    occluded_image = image.copy()

    draw = ImageDraw.Draw(occluded_image)

    draw.rectangle(
        (left, top, right - 1, bottom - 1),
        fill=OCCLUSION_COLOR,
    )

    return occluded_image


# ============================================================
# RUN IMAGE OCCLUSION EXPERIMENT
# ============================================================

def run_experiment(
    model,
    processor,
    vision_model,
    scaler_mean,
    scaler_scale,
    classes,
    test_rows,
):

    conditions = [
        "original",
        "central_occlusion",
    ]

    all_embeddings = {
        condition: []
        for condition in conditions
    }

    labels = []
    filenames = []
    failures = []

    for index, row in enumerate(test_rows, start=1):

        category = row["category"]
        filename = row["filename"]

        video_path = DATA_DIR / category / filename

        try:

            frames = sample_video_frames(
                video_path,
                NUM_FRAMES,
            )

            if frames is None:
                raise ValueError(
                    "Unable to sample eight frames."
                )

            transformed_frames = {
                "original": frames,

                "central_occlusion": [
                    apply_central_occlusion(frame)
                    for frame in frames
                ],
            }

            for condition in conditions:

                video_embeddings = extract_frame_embeddings(
                    transformed_frames[condition],
                    processor,
                    vision_model,
                    DEVICE,
                )

                if video_embeddings.shape != (
                    NUM_FRAMES,
                    len(scaler_mean),
                ):
                    raise ValueError(
                        f"Unexpected embedding shape: "
                        f"{video_embeddings.shape}"
                    )

                all_embeddings[condition].append(
                    video_embeddings
                )

            labels.append(category)
            filenames.append(filename)

        except Exception as error:

            failures.append({
                "category": category,
                "filename": filename,
                "error": str(error),
            })

            print(
                f"\nFailed: {filename} - {error}"
            )

        print(
            f"\rProcessed videos: {index}/{len(test_rows)} | "
            f"Successful: {len(labels)} | "
            f"Failed: {len(failures)}",
            end="",
            flush=True,
        )

    print()

    if failures:

        failure_path = RESULTS_DIR / "failures.json"

        with failure_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(failures, file, indent=4)

        raise RuntimeError(
            f"{len(failures)} videos failed. "
            f"See {failure_path}. "
            "Do not compare incomplete conditions."
        )

    if len(labels) != EXPECTED_TEST_VIDEOS:
        raise RuntimeError(
            "Not all official test videos were evaluated."
        )

    results = {}

    for condition in conditions:

        print(f"\nEvaluating condition: {condition}")

        metrics = evaluate_gru_predictions(
            model=model,
            embeddings=all_embeddings[condition],
            labels=labels,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            classes=classes,
            device=DEVICE,
            num_frames=NUM_FRAMES,
        )

        results[condition] = {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "classification_report": metrics[
                "classification_report"
            ],
            "confusion_matrix": metrics["confusion_matrix"],
            "predictions": metrics["predictions"].tolist(),
        }

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

    original_accuracy = results["original"]["accuracy"]
    original_f1 = results["original"]["macro_f1"]

    for condition in conditions:

        results[condition]["accuracy_change"] = (
            results[condition]["accuracy"] - original_accuracy
        )

        results[condition]["macro_f1_change"] = (
            results[condition]["macro_f1"] - original_f1
        )

    return results, labels, filenames


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, labels, filenames, classes):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prediction_rows = []

    for condition, metrics in results.items():

        predictions = metrics["predictions"]

        for filename, true_label, prediction_index in zip(
            filenames,
            labels,
            predictions,
        ):

            prediction_rows.append({
                "condition": condition,
                "filename": filename,
                "true_label": true_label,
                "predicted_label": classes[prediction_index],
                "correct": (
                    true_label == classes[prediction_index]
                ),
            })

    pd.DataFrame(prediction_rows).to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    metric_rows = []

    for condition, metrics in results.items():

        metric_rows.append({
            "condition": condition,
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "accuracy_change": metrics["accuracy_change"],
            "macro_f1_change": metrics["macro_f1_change"],
        })

    metrics_df = pd.DataFrame(metric_rows)

    metrics_df.to_csv(
        RESULTS_DIR / "metrics.csv",
        index=False,
    )

    json_results = {}

    for condition, metrics in results.items():

        json_results[condition] = {
            key: value
            for key, value in metrics.items()
            if key != "predictions"
        }

    output = {
        "experiment": "image_level_central_occlusion_robustness",
        "dataset": "HMDB51 selected six classes",
        "test_videos": len(labels),
        "frames_per_video": NUM_FRAMES,
        "embedding_model": MODEL_NAME,
        "classifier": "trained DINOv2 temporal GRU",
        "classifier_checkpoint": str(
            CHECKPOINT_FILE.relative_to(PROJECT_ROOT)
        ),
        "device": str(DEVICE),
        "occlusion": {
            "shape": "central_rectangle",
            "width_ratio": OCCLUSION_WIDTH_RATIO,
            "height_ratio": OCCLUSION_HEIGHT_RATIO,
            "color_rgb": OCCLUSION_COLOR,
            "applied_to": "all sampled frames",
        },
        "results": json_results,
    }

    with (
        RESULTS_DIR / "image_occlusion_metrics.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(output, file, indent=4)

    # Accuracy and macro-F1 comparison.
    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.bar(
        x - width / 2,
        metrics_df["accuracy"] * 100,
        width,
        label="Accuracy",
    )

    ax.bar(
        x + width / 2,
        metrics_df["macro_f1"] * 100,
        width,
        label="Macro-F1",
    )

    ax.set_ylabel("Score (%)")
    ax.set_title("GRU Performance Under Central Image Occlusion")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["condition"])

    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "image_occlusion_comparison.png",
        dpi=300,
    )

    plt.close(fig)

    # Accuracy change relative to original frames.
    fig, ax = plt.subplots(figsize=(8, 5))

    occlusion_change = metrics_df[
        metrics_df["condition"] != "original"
    ]

    ax.bar(
        occlusion_change["condition"],
        occlusion_change["accuracy_change"] * 100,
    )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("Accuracy Change After Central Occlusion")

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "image_occlusion_accuracy_change.png",
        dpi=300,
    )

    plt.close(fig)

    print("\n========== IMAGE OCCLUSION SUMMARY ==========")

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print(RESULTS_DIR / "image_occlusion_metrics.json")
    print(RESULTS_DIR / "metrics.csv")
    print(RESULTS_DIR / "predictions.csv")
    print(PLOT_DIR)


# ============================================================
# MAIN
# ============================================================

def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading trained GRU...")

    (
        model,
        scaler_mean,
        scaler_scale,
        classes,
        checkpoint,
    ) = load_temporal_gru(CHECKPOINT_FILE, DEVICE)

    if checkpoint["num_frames"] != NUM_FRAMES:
        raise ValueError(
            "GRU checkpoint frame count does not match extraction."
        )

    if checkpoint.get("model_name") not in (
        None,
        MODEL_NAME,
    ):
        raise ValueError(
            "Embedding model does not match the GRU checkpoint."
        )

    print("\nLoading DINOv2 processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    print("Loading pretrained DINOv2 model...")

    vision_model = AutoModel.from_pretrained(
        MODEL_NAME
    )

    vision_model.to(DEVICE)
    vision_model.eval()

    for parameter in vision_model.parameters():
        parameter.requires_grad = False

    test_rows = load_test_inventory()

    print("\n========== EXPERIMENT CONFIGURATION ==========")
    print(f"Test videos: {len(test_rows)}")
    print(f"Frames per video: {NUM_FRAMES}")
    print(f"DINOv2: {MODEL_NAME}")
    print(
        "Occlusion width: "
        f"{OCCLUSION_WIDTH_RATIO * 100:.0f}%"
    )
    print(
        "Occlusion height: "
        f"{OCCLUSION_HEIGHT_RATIO * 100:.0f}%"
    )
    print(f"Occlusion color: {OCCLUSION_COLOR}")
    print(f"Device: {DEVICE}")

    results, labels, filenames = run_experiment(
        model,
        processor,
        vision_model,
        scaler_mean,
        scaler_scale,
        classes,
        test_rows,
    )

    save_results(
        results,
        labels,
        filenames,
        classes,
    )

    print("\nImage occlusion experiment completed successfully.")


if __name__ == "__main__":
    main()
