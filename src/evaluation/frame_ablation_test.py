import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, f1_score


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_FILE = (
    PROJECT_ROOT / "data/processed/dinov2_hmdb51_split1.npz"
)

CHECKPOINT_FILE = PROJECT_ROOT / "results/gru_temporal.pth"

RESULTS_DIR = PROJECT_ROOT / "results/frame_ablation"
PLOT_DIR = RESULTS_DIR / "plots"

JSON_FILE = RESULTS_DIR / "frame_ablation_metrics.json"
CSV_FILE = RESULTS_DIR / "frame_ablation_metrics.csv"
PLOT_FILE = PLOT_DIR / "frame_ablation_accuracy.png"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL ARCHITECTURE
# Must match the trained GRU.
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
# LOAD MODEL AND CHECKPOINT SCALER
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

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    scaler_mean = checkpoint["scaler_mean"].astype(np.float32)
    scaler_scale = checkpoint["scaler_scale"].astype(np.float32)
    classes = np.asarray(checkpoint["classes"])

    print("GRU checkpoint loaded.")
    print(f"Device: {DEVICE}")
    print(f"Classes: {classes.tolist()}")
    print(f"Frames per video: {checkpoint['num_frames']}")

    return model, scaler_mean, scaler_scale, classes, checkpoint


# ============================================================
# LOAD OFFICIAL TEST SPLIT
# ============================================================

def load_test_data(scaler_mean, scaler_scale, checkpoint):

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
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
    y_test_labels = labels[test_indices]

    # Use the exact scaler saved during GRU training.
    X_test = (
        X_test - scaler_mean.reshape(1, 1, -1)
    ) / scaler_scale.reshape(1, 1, -1)

    X_test = X_test.astype(np.float32)

    class_to_index = {
        class_name: index
        for index, class_name in enumerate(checkpoint["classes"])
    }

    unknown_classes = (
        set(y_test_labels.tolist()) - set(class_to_index.keys())
    )

    if unknown_classes:
        raise ValueError(
            f"Unknown test classes: {sorted(unknown_classes)}"
        )

    y_test = np.asarray(
        [class_to_index[label] for label in y_test_labels],
        dtype=np.int64
    )

    print(f"Test videos: {len(X_test)}")
    print(f"Test sequence shape: {X_test.shape}")

    return X_test, y_test


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, X_test, y_test):

    predictions = []

    batch_size = 32

    with torch.no_grad():
        for start in range(0, len(X_test), batch_size):

            batch = X_test[start:start + batch_size]

            batch_tensor = torch.from_numpy(batch).to(DEVICE)

            logits = model(batch_tensor)

            batch_predictions = (
                torch.argmax(logits, dim=1)
                .cpu()
                .numpy()
            )

            predictions.extend(batch_predictions.tolist())

    accuracy = accuracy_score(y_test, predictions)

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1)
    }


# ============================================================
# FRAME-ABLATION EXPERIMENT
# ============================================================

def run_experiment(model, X_test, y_test):

    results = {}

    # Reference: all frames retained.
    print("\nEvaluating original sequences...")

    original_metrics = evaluate_model(
        model,
        X_test,
        y_test
    )

    results["original"] = {
        "condition": "original",
        "frame_removed": None,
        **original_metrics
    }

    print(
        f"Original: accuracy={original_metrics['accuracy']:.4f}, "
        f"macro-F1={original_metrics['macro_f1']:.4f}"
    )

    num_frames = X_test.shape[1]

    for frame_index in range(num_frames):

        print(
            f"\nReplacing frame {frame_index} "
            f"of {num_frames - 1}..."
        )

        # Copy so the original data remains unchanged.
        X_ablated = X_test.copy()

        # Standardized zero = training-set mean embedding.
        X_ablated[:, frame_index, :] = 0.0

        metrics = evaluate_model(
            model,
            X_ablated,
            y_test
        )

        condition_name = f"frame_{frame_index}_ablated"

        results[condition_name] = {
            "condition": condition_name,
            "frame_removed": frame_index,
            **metrics,
            "accuracy_change": (
                metrics["accuracy"] - original_metrics["accuracy"]
            ),
            "macro_f1_change": (
                metrics["macro_f1"] - original_metrics["macro_f1"]
            )
        }

        print(
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Macro-F1: {metrics['macro_f1']:.4f} | "
            f"Accuracy change: "
            f"{results[condition_name]['accuracy_change']:+.4f}"
        )

    return results


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results):

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output.
    output = {
        "experiment": "single-frame embedding ablation",
        "description": (
            "Each standardized frame embedding is replaced individually "
            "with a zero vector. Because features are standardized, zero "
            "corresponds to the training-set mean embedding."
        ),
        "results": results
    }

    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    # CSV output.
    df = pd.DataFrame(list(results.values()))
    df.to_csv(CSV_FILE, index=False)

    # Accuracy plot.
    frame_results = df[df["frame_removed"].notna()].copy()

    frame_results["frame_removed"] = (
        frame_results["frame_removed"].astype(int)
    )

    original_accuracy = results["original"]["accuracy"]

    plt.figure(figsize=(10, 5))

    plt.bar(
        frame_results["frame_removed"],
        frame_results["accuracy"]
    )

    plt.axhline(
        y=original_accuracy,
        linestyle="--",
        label=f"Original accuracy ({original_accuracy:.3f})"
    )

    plt.xticks(range(8))
    plt.xlabel("Frame index replaced by mean embedding")
    plt.ylabel("Test accuracy")
    plt.title("GRU Sensitivity to Individual Frame Ablation")
    plt.legend()
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=200)
    plt.close()

    # Console summary.
    print("\n========== FRAME ABLATION SUMMARY ==========")

    print(
        f"Original accuracy: "
        f"{original_accuracy:.4f}"
    )

    print(
        f"Original macro-F1: "
        f"{results['original']['macro_f1']:.4f}"
    )

    print("\nCondition              Accuracy    Macro-F1    Accuracy change")

    print(
        f"{'original':<22}"
        f"{results['original']['accuracy']:<12.4f}"
        f"{results['original']['macro_f1']:<12.4f}"
        f"{'—'}"
    )

    for _, row in frame_results.iterrows():

        print(
            f"{row['condition']:<22}"
            f"{row['accuracy']:<12.4f}"
            f"{row['macro_f1']:<12.4f}"
            f"{row['accuracy_change']:+.4f}"
        )

    print(f"\nJSON saved to: {JSON_FILE}")
    print(f"CSV saved to: {CSV_FILE}")
    print(f"Plot saved to: {PLOT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():

    model, scaler_mean, scaler_scale, classes, checkpoint = load_model()

    X_test, y_test = load_test_data(
        scaler_mean,
        scaler_scale,
        checkpoint
    )

    results = run_experiment(
        model,
        X_test,
        y_test
    )

    save_results(results)


if __name__ == "__main__":
    main()