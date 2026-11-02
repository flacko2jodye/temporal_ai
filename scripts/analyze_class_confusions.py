from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dinov2_hmdb51_split1.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = RESULTS_DIR / "class_confusion_analysis"

CHECKPOINTS = {
    "single_frame": RESULTS_DIR / "single_frame_baseline.pth",
    "mean_pool": RESULTS_DIR / "mean_pool_baseline.pth",
    "gru": RESULTS_DIR / "gru_temporal.pth",
}

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL ARCHITECTURES
# Reconstructed from the original training scripts.
# ============================================================

class SingleFrameClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class MeanPoolClassifier(nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.network(x)


class TemporalGRU(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.3,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        _, hidden = self.gru(x)
        final_hidden = hidden[-1]
        return self.classifier(final_hidden)


# ============================================================
# DATA LOADING
# ============================================================

def load_test_data():
    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        embeddings = data["embeddings"].astype(np.float32)
        labels = data["labels"]
        splits = data["splits"]
        filenames = data["filenames"]

    test_mask = splits == "test"

    test_embeddings = embeddings[test_mask]
    test_labels = labels[test_mask]
    test_filenames = filenames[test_mask]

    if len(test_labels) != 180:
        raise ValueError(
            f"Expected 180 official test videos, "
            f"found {len(test_labels)}."
        )

    classes = sorted(np.unique(labels).tolist())

    print("=" * 72)
    print("DATASET")
    print("=" * 72)
    print(f"Embedding file: {EMBEDDING_FILE}")
    print(f"All videos: {len(embeddings)}")
    print(f"Official test videos: {len(test_labels)}")
    print(f"Embedding shape: {test_embeddings.shape}")
    print(f"Classes: {classes}")

    return (
        test_embeddings,
        test_labels,
        test_filenames,
        classes,
    )


# ============================================================
# CHECKPOINT LOADING
# ============================================================

def load_checkpoint(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}"
        )

    # weights_only=False is used because these are the user's
    # own locally generated checkpoints containing metadata.
    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    required_keys = {
        "model_state_dict",
        "classes",
        "scaler_mean",
        "scaler_scale",
        "input_dim",
        "model_type",
    }

    missing = required_keys - set(checkpoint.keys())

    if missing:
        raise ValueError(
            f"Checkpoint {path.name} is missing keys: {missing}"
        )

    return checkpoint


# ============================================================
# MODEL RECONSTRUCTION
# ============================================================

def build_model(model_name, checkpoint, num_classes):
    input_dim = int(checkpoint["input_dim"])

    if model_name == "single_frame":
        model = SingleFrameClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    elif model_name == "mean_pool":
        model = MeanPoolClassifier(
            input_dim=input_dim,
            num_classes=num_classes,
        )

    elif model_name == "gru":
        model = TemporalGRU(
            input_dim=input_dim,
            hidden_dim=int(checkpoint["hidden_dim"]),
            num_layers=int(checkpoint["num_layers"]),
            num_classes=num_classes,
            dropout=0.3,
        )

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model


# ============================================================
# PREPROCESSING AND INFERENCE
# ============================================================

def predict_model(
    model_name,
    checkpoint,
    test_embeddings,
    classes,
):
    model = build_model(
        model_name,
        checkpoint,
        num_classes=len(classes),
    )

    scaler_mean = np.asarray(
        checkpoint["scaler_mean"],
        dtype=np.float32,
    )

    scaler_scale = np.asarray(
        checkpoint["scaler_scale"],
        dtype=np.float32,
    )

    if np.any(scaler_scale == 0):
        raise ValueError(
            f"{model_name}: scaler contains zero scales."
        )

    if model_name == "single_frame":
        # Match original training: select frame index 4,
        # then scale the selected embedding.
        selected = test_embeddings[:, 4, :]

        inputs = (
            selected - scaler_mean
        ) / scaler_scale

    elif model_name == "mean_pool":
        # Match original training: mean-pool first,
        # then apply the fitted scaler.
        pooled = test_embeddings.mean(axis=1)

        inputs = (
            pooled - scaler_mean
        ) / scaler_scale

    else:
        # Match original training: scale each frame
        # independently, preserving the sequence order.
        inputs = (
            test_embeddings - scaler_mean[None, None, :]
        ) / scaler_scale[None, None, :]

    inputs = np.asarray(inputs, dtype=np.float32)

    with torch.no_grad():
        tensor = torch.from_numpy(inputs).to(DEVICE)
        logits = model(tensor)
        predictions = logits.argmax(dim=1).cpu().numpy()

    del model

    if not np.isfinite(inputs).all():
        raise ValueError(
            f"{model_name}: preprocessing produced non-finite values."
        )

    return predictions


# ============================================================
# CONFUSION ANALYSIS
# ============================================================

def analyze_model(
    model_name,
    checkpoint_path,
    test_embeddings,
    test_labels,
    test_filenames,
    classes,
):
    print("\n" + "=" * 72)
    print(f"MODEL: {model_name}")
    print("=" * 72)

    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_classes = list(checkpoint["classes"])

    if checkpoint_classes != classes:
        raise ValueError(
            f"Class ordering mismatch in {checkpoint_path.name}.\n"
            f"Checkpoint: {checkpoint_classes}\n"
            f"Dataset: {classes}"
        )

    # Model predictions are numeric indices (0..5).
    predicted_indices = predict_model(
        model_name,
        checkpoint,
        test_embeddings,
        classes,
    )

    # Convert predictions to class names so they match test_labels.
    predictions = np.asarray(
        [classes[int(index)] for index in predicted_indices]
    )

    accuracy = accuracy_score(test_labels, predictions)

    macro_f1 = f1_score(
        test_labels,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        test_labels,
        predictions,
        labels=classes,
    )

    print(f"Test videos: {len(test_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nConfusion matrix")
    print("Rows = true classes; columns = predicted classes")
    print(
        pd.DataFrame(
            matrix,
            index=classes,
            columns=classes,
        ).to_string()
    )

    print("\nPer-class classification report")

    report = classification_report(
        test_labels,
        predictions,
        labels=classes,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).T

    print(
        report_df.loc[
            classes,
            ["precision", "recall", "f1-score", "support"],
        ].round(4).to_string()
    )

    # Find the largest off-diagonal confusion counts.
    confusion_pairs = []

    for true_idx, true_class in enumerate(classes):
        for pred_idx, pred_class in enumerate(classes):
            if true_idx == pred_idx:
                continue

            count = int(matrix[true_idx, pred_idx])

            if count > 0:
                confusion_pairs.append(
                    {
                        "true_class": true_class,
                        "predicted_class": pred_class,
                        "count": count,
                    }
                )

    confusion_pairs.sort(
        key=lambda item: item["count"],
        reverse=True,
    )

    print("\nMost frequent class confusions")

    if not confusion_pairs:
        print("No off-diagonal errors.")
    else:
        for item in confusion_pairs[:10]:
            print(
                f"  {item['true_class']:12s} -> "
                f"{item['predicted_class']:12s}: "
                f"{item['count']}"
            )

    prediction_df = pd.DataFrame(
        {
            "filename": test_filenames,
            "true_label": test_labels,
            "predicted_label": predictions,
            "correct": test_labels == predictions,
        }
    )

    errors_df = prediction_df[
        ~prediction_df["correct"]
    ].copy()

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
        "confusion_pairs": confusion_pairs,
        "predictions": prediction_df,
        "errors": errors_df,
    }


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(results, classes):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {}

    for result in results:
        name = result["model"]

        summary[name] = {
            "accuracy": result["accuracy"],
            "macro_f1": result["macro_f1"],
            "confusion_matrix": result["confusion_matrix"],
            "classification_report": result["classification_report"],
            "top_confusions": result["confusion_pairs"][:10],
        }

        result["predictions"].to_csv(
            OUTPUT_DIR / f"{name}_predictions.csv",
            index=False,
        )

        result["errors"].to_csv(
            OUTPUT_DIR / f"{name}_errors.csv",
            index=False,
        )

        pd.DataFrame(
            result["confusion_matrix"],
            index=classes,
            columns=classes,
        ).to_csv(
            OUTPUT_DIR / f"{name}_confusion_matrix.csv"
        )

    with (OUTPUT_DIR / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=2)

    comparison = pd.DataFrame(
        [
            {
                "model": result["model"],
                "accuracy": result["accuracy"],
                "macro_f1": result["macro_f1"],
                "misclassified_videos": len(result["errors"]),
            }
            for result in results
        ]
    )

    comparison.to_csv(
        OUTPUT_DIR / "model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 72)
    print("MODEL COMPARISON")
    print("=" * 72)
    print(comparison.round(4).to_string(index=False))

    print("\nSaved analysis artifacts:")
    print(f"Directory: {OUTPUT_DIR}")

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {path.name}")


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Device: {DEVICE}")

    (
        test_embeddings,
        test_labels,
        test_filenames,
        classes,
    ) = load_test_data()

    results = []

    for model_name, checkpoint_path in CHECKPOINTS.items():
        result = analyze_model(
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            test_embeddings=test_embeddings,
            test_labels=test_labels,
            test_filenames=test_filenames,
            classes=classes,
        )

        results.append(result)

    save_results(results, classes)


if __name__ == "__main__":
    main()