
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from PIL import Image, ImageEnhance, ImageFilter
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

RESULTS_DIR = PROJECT_ROOT / "results/visual_corruption"
PLOT_DIR = RESULTS_DIR / "plots"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RANDOM_SEED = 42

# Corruption parameters.
BLUR_RADIUS = 2.0
NOISE_STD = 20.0
BRIGHTNESS_FACTOR = 0.5

# Keep this consistent with the original extraction pipeline.
EXPECTED_TEST_VIDEOS = 180


# ============================================================
# IMAGE CORRUPTIONS
# Input and output are PIL RGB images.
# ============================================================

def apply_gaussian_blur(image):

    return image.filter(
        ImageFilter.GaussianBlur(radius=BLUR_RADIUS)
    )


def apply_gaussian_noise(image, rng):

    image_array = np.asarray(
        image,
        dtype=np.float32,
    )

    noise = rng.normal(
        loc=0.0,
        scale=NOISE_STD,
        size=image_array.shape,
    )

    noisy_image = np.clip(
        image_array + noise,
        0,
        255,
    ).astype(np.uint8)

    return Image.fromarray(noisy_image)


def apply_brightness_reduction(image):

    return ImageEnhance.Brightness(image).enhance(
        BRIGHTNESS_FACTOR
    )


# ============================================================
# RUN VISUAL CORRUPTION EXPERIMENT
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
        "gaussian_blur",
        "gaussian_noise",
        "brightness_reduction",
    ]

    all_embeddings = {
        condition: []
        for condition in conditions
    }

    labels = []
    filenames = []

    rng = np.random.default_rng(RANDOM_SEED)

    failures = []

    for index, row in enumerate(test_rows, start=1):

        category = row["category"]
        filename = row["filename"]

        video_path = DATA_DIR / category / filename

        try:

            # Original frame sampling is reused exactly.
            frames = sample_video_frames(
                video_path,
                NUM_FRAMES,
            )

            if frames is None:
                raise ValueError(
                    "Unable to sample eight frames."
                )

            # Apply each transformation to copies of the
            # same eight sampled RGB frames.
            transformed_frames = {
                "original": frames,

                "gaussian_blur": [
                    apply_gaussian_blur(frame)
                    for frame in frames
                ],

                "gaussian_noise": [
                    apply_gaussian_noise(frame, rng)
                    for frame in frames
                ],

                "brightness_reduction": [
                    apply_brightness_reduction(frame)
                    for frame in frames
                ],
            }

            # Extract embeddings independently for each condition.
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

    # Evaluate every condition with the same frozen GRU.
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
        }

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f}"
        )

        results[condition]["predictions"] = (
            metrics["predictions"].tolist()
        )

    # Add changes relative to the original condition.
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

    # Save predictions for every video and condition.
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

    prediction_df = pd.DataFrame(prediction_rows)

    prediction_df.to_csv(
        RESULTS_DIR / "predictions.csv",
        index=False,
    )

    # Save compact metrics table.
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

    # Save full JSON report without duplicating predictions.
    json_results = {}

    for condition, metrics in results.items():

        json_results[condition] = {
            key: value
            for key, value in metrics.items()
            if key != "predictions"
        }

    output = {
        "experiment": "image_level_visual_corruption_robustness",
        "dataset": "HMDB51 selected six classes",
        "test_videos": len(labels),
        "frames_per_video": NUM_FRAMES,
        "embedding_model": MODEL_NAME,
        "classifier": "trained DINOv2 temporal GRU",
        "classifier_checkpoint": str(
            CHECKPOINT_FILE.relative_to(PROJECT_ROOT)
        ),
        "device": str(DEVICE),
        "random_seed": RANDOM_SEED,
        "transformations": {
            "original": {
                "description": "No image transformation",
            },
            "gaussian_blur": {
                "radius": BLUR_RADIUS,
            },
            "gaussian_noise": {
                "standard_deviation": NOISE_STD,
                "pixel_range": "0-255",
                "seed": RANDOM_SEED,
            },
            "brightness_reduction": {
                "factor": BRIGHTNESS_FACTOR,
            },
        },
        "results": json_results,
    }

    with (
        RESULTS_DIR / "visual_corruption_metrics.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(output, file, indent=4)

    # Comparison plot.
    x = np.arange(len(metrics_df))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

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
    ax.set_title("GRU Robustness Under Visual Corruption")

    ax.set_xticks(x)
    ax.set_xticklabels(
        metrics_df["condition"],
        rotation=15,
        ha="right",
    )

    ax.set_ylim(0, 100)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "visual_corruption_comparison.png",
        dpi=300,
    )

    plt.close(fig)

    # Accuracy degradation plot.
    fig, ax = plt.subplots(figsize=(9, 5))

    corruption_df = metrics_df[
        metrics_df["condition"] != "original"
    ]

    ax.bar(
        corruption_df["condition"],
        corruption_df["accuracy_change"] * 100,
    )

    ax.axhline(0, linestyle="--")

    ax.set_ylabel("Accuracy change (percentage points)")
    ax.set_title("Accuracy Change Relative to Original Videos")

    ax.tick_params(
        axis="x",
        rotation=15,
    )

    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        PLOT_DIR / "visual_corruption_accuracy_change.png",
        dpi=300,
    )

    plt.close(fig)

    print("\n========== VISUAL CORRUPTION SUMMARY ==========")

    print(
        metrics_df.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print("\nSaved outputs:")
    print(RESULTS_DIR / "visual_corruption_metrics.json")
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
    print(f"Blur radius: {BLUR_RADIUS}")
    print(f"Noise standard deviation: {NOISE_STD}")
    print(f"Brightness factor: {BRIGHTNESS_FACTOR}")
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

    print("\nVisual corruption experiment completed successfully.")


if __name__ == "__main__":
    main()
