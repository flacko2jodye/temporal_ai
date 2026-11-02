
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_FILE = (
    PROJECT_ROOT / "data/processed/dinov2_hmdb51_split1.npz"
)

CHECKPOINT_FILE = PROJECT_ROOT / "results/gru_temporal.pth"

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_FILE = RESULTS_DIR / "temporal_order_metrics.json"

RANDOM_STATE = 42
NUM_SHUFFLES = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# Must match the original GRU training architecture.
# ============================================================

class TemporalGRU(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.3
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        _, hidden = self.gru(x)

        final_hidden = hidden[-1]

        return self.classifier(final_hidden)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    if not CHECKPOINT_FILE.exists():
        raise FileNotFoundError(
            f"GRU checkpoint not found: {CHECKPOINT_FILE}"
        )

    checkpoint = torch.load(
        CHECKPOINT_FILE,
        map_location="cpu",
        weights_only=False
    )

    model = TemporalGRU(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(checkpoint["classes"])
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    scaler_mean = checkpoint["scaler_mean"].astype(np.float32)
    scaler_scale = checkpoint["scaler_scale"].astype(np.float32)

    classes = np.asarray(checkpoint["classes"])

    print("GRU checkpoint loaded.")
    print(f"Model: {checkpoint['model_type']}")
    print(f"Classes: {list(classes)}")
    print(f"Frames per video: {checkpoint['num_frames']}")
    print(f"Embedding dimension: {checkpoint['input_dim']}")

    return model, scaler_mean, scaler_scale, classes, checkpoint


# ============================================================
# LOAD TEST DATA
# Reproduce the original official train/test split.
# ============================================================

def load_test_data(
    scaler_mean,
    scaler_scale,
    checkpoint
):

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with np.load(
        EMBEDDING_FILE,
        allow_pickle=False
    ) as data:

        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"]
        splits = data["splits"]

    if embeddings.ndim != 3:
        raise ValueError(
            f"Expected 3D embeddings, got {embeddings.shape}"
        )

    if embeddings.shape[1] != checkpoint["num_frames"]:
        raise ValueError("Frame count does not match checkpoint.")

    if embeddings.shape[2] != checkpoint["input_dim"]:
        raise ValueError("Embedding dimension does not match checkpoint.")

    test_indices = np.where(splits == "test")[0]

    X_test = embeddings[test_indices]
    y_test = labels[test_indices]

    # Apply the exact training scaler from the checkpoint.
    X_test = (
        X_test - scaler_mean.reshape(1, 1, -1)
    ) / scaler_scale.reshape(1, 1, -1)

    X_test = X_test.astype(np.float32)

    classes = np.asarray(checkpoint["classes"])

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(classes)
    }

    unknown_classes = sorted(
        set(y_test.tolist()) - set(class_to_index.keys())
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown test labels: {unknown_classes}"
        )

    y_test = np.asarray(
        [class_to_index[label] for label in y_test],
        dtype=np.int64
    )

    print("\n========== TEST DATA SUMMARY ==========")
    print(f"Test videos: {len(X_test)}")
    print(f"Sequence shape: {X_test.shape}")
    print(f"Classes: {list(classes)}")

    return X_test, y_test, classes


# ============================================================
# SEQUENCE ORDER TRANSFORMATIONS
# ============================================================

def create_order_conditions(X_test):
    rng = np.random.default_rng(RANDOM_STATE)

    num_frames = X_test.shape[1]
    original_order = np.arange(num_frames)
    reversed_order = original_order[::-1]

    conditions = {
        "original": X_test,
        "reversed": X_test[:, ::-1, :].copy(),
    }

    permutations = []
    seen_permutations = set()

    while len(permutations) < NUM_SHUFFLES:
        permutation = rng.permutation(num_frames)
        permutation_tuple = tuple(permutation.tolist())

        # Exclude the original and reversed sequences.
        if np.array_equal(permutation, original_order):
            continue

        if np.array_equal(permutation, reversed_order):
            continue

        # Avoid evaluating the same permutation twice.
        if permutation_tuple in seen_permutations:
            continue

        seen_permutations.add(permutation_tuple)
        permutations.append(permutation)

    print("\n========== ORDER CONDITIONS ==========")
    print(f"Original order: {original_order.tolist()}")
    print(f"Reversed order: {reversed_order.tolist()}")

    for index, permutation in enumerate(permutations, start=1):
        condition_name = f"shuffled_{index:02d}"

        conditions[condition_name] = (
            X_test[:, permutation, :].copy()
        )

        print(
            f"{condition_name}: {permutation.tolist()}"
        )

    return conditions, [
        permutation.tolist()
        for permutation in permutations
    ]


# ============================================================
# EVALUATION
# ============================================================

def evaluate_condition(
    model,
    X_test,
    y_test,
    classes,
    condition_name
):

    model.eval()

    predictions = []

    batch_size = 32

    with torch.no_grad():

        for start in range(0, len(X_test), batch_size):

            batch = torch.tensor(
                X_test[start:start + batch_size],
                dtype=torch.float32,
                device=DEVICE
            )

            outputs = model(batch)

            batch_predictions = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                batch_predictions.cpu().numpy()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.int64
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=np.arange(len(classes))
    )

    print(f"\n========== {condition_name.upper()} ==========")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=np.arange(len(classes)),
            target_names=classes,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(matrix)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "predictions": predictions.tolist()
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, permutations, classes):
    original_accuracy = results["original"]["accuracy"]
    original_f1 = results["original"]["macro_f1"]

    shuffled_names = sorted(
        name
        for name in results
        if name.startswith("shuffled_")
    )

    shuffled_accuracies = np.array(
        [
            results[name]["accuracy"]
            for name in shuffled_names
        ],
        dtype=np.float64,
    )

    shuffled_f1_scores = np.array(
        [
            results[name]["macro_f1"]
            for name in shuffled_names
        ],
        dtype=np.float64,
    )

    output = {
        "experiment": "temporal_order_sensitivity",
        "model": "temporal_gru",
        "test_samples": len(results["original"]["predictions"]),
        "frames_per_video": 8,
        "embedding_dimension": 384,
        "random_state": RANDOM_STATE,
        "number_of_shuffles": len(shuffled_names),
        "classes": classes.tolist(),
        "conditions": {
            name: {
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "confusion_matrix": result["confusion_matrix"],
            }
            for name, result in results.items()
        },
        "shuffled_frame_permutations": {
            name: permutation
            for name, permutation in zip(
                shuffled_names,
                permutations,
            )
        },
        "change_from_original": {
            name: {
                "accuracy_difference": (
                    result["accuracy"] - original_accuracy
                ),
                "macro_f1_difference": (
                    result["macro_f1"] - original_f1
                ),
            }
            for name, result in results.items()
            if name != "original"
        },
        "shuffle_statistics": {
            "accuracy_mean": float(shuffled_accuracies.mean()),
            "accuracy_std": float(shuffled_accuracies.std(ddof=1)),
            "accuracy_min": float(shuffled_accuracies.min()),
            "accuracy_max": float(shuffled_accuracies.max()),
            "macro_f1_mean": float(shuffled_f1_scores.mean()),
            "macro_f1_std": float(shuffled_f1_scores.std(ddof=1)),
            "macro_f1_min": float(shuffled_f1_scores.min()),
            "macro_f1_max": float(shuffled_f1_scores.max()),
            "accuracy_difference_from_original": float(
                shuffled_accuracies.mean() - original_accuracy
            ),
            "macro_f1_difference_from_original": float(
                shuffled_f1_scores.mean() - original_f1
            ),
            "shuffles_below_original_accuracy": int(
                np.sum(shuffled_accuracies < original_accuracy)
            ),
            "shuffles_below_original_macro_f1": int(
                np.sum(shuffled_f1_scores < original_f1)
            ),
        },
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\n========== ORDER SENSITIVITY RESULTS ==========")
    print(f"{'Condition':<15}{'Accuracy':<15}{'Macro F1':<15}")

    for name, result in results.items():
        print(
            f"{name:<15}"
            f"{result['accuracy']:<15.4f}"
            f"{result['macro_f1']:<15.4f}"
        )

    stats = output["shuffle_statistics"]

    print("\n========== SHUFFLE SUMMARY ==========")
    print(
        f"Original accuracy:       {original_accuracy:.4f}"
    )
    print(
        f"Shuffled accuracy:       "
        f"{stats['accuracy_mean']:.4f} "
        f"+/- {stats['accuracy_std']:.4f}"
    )
    print(
        f"Original macro-F1:       {original_f1:.4f}"
    )
    print(
        f"Shuffled macro-F1:       "
        f"{stats['macro_f1_mean']:.4f} "
        f"+/- {stats['macro_f1_std']:.4f}"
    )

    print(
        f"\nShuffles below original accuracy: "
        f"{stats['shuffles_below_original_accuracy']}"
        f"/{len(shuffled_names)}"
    )

    print(
        f"Shuffles below original macro-F1: "
        f"{stats['shuffles_below_original_macro_f1']}"
        f"/{len(shuffled_names)}"
    )

    print("\nChanges relative to original order:")

    for name, difference in output["change_from_original"].items():
        print(
            f"{name}: "
            f"Accuracy {difference['accuracy_difference']:+.4f} | "
            f"Macro F1 {difference['macro_f1_difference']:+.4f}"
        )

    print(f"\nResults saved to: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Using device: {DEVICE}")

    model, scaler_mean, scaler_scale, classes, checkpoint = (
        load_model()
    )

    X_test, y_test, classes = load_test_data(
        scaler_mean,
        scaler_scale,
        checkpoint,
    )

    conditions, permutations = create_order_conditions(X_test)

    results = {}

    for condition_name, sequences in conditions.items():
        results[condition_name] = evaluate_condition(
            model,
            sequences,
            y_test,
            classes,
            condition_name,
        )

    save_results(
        results,
        permutations,
        classes,
    )


if __name__ == "__main__":
    main()
