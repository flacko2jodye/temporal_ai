
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_FILE = (
    PROJECT_ROOT
    / "data/processed/dinov2_hmdb51_split1.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

RANDOM_STATE = 42

BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3
PATIENCE = 8

INPUT_DIM = 384
HIDDEN_DIM = 128
NUM_LAYERS = 1
DROPOUT = 0.3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TEMPORAL GRU CLASSIFIER
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

        # x shape: (batch, sequence_length, input_dim)
        _, hidden = self.gru(x)

        # Final hidden state of the GRU.
        final_hidden = hidden[-1]

        return self.classifier(final_hidden)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data():

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

    num_videos, num_frames, embedding_dim = embeddings.shape

    if embedding_dim != INPUT_DIM:
        raise ValueError(
            f"Expected embedding dimension {INPUT_DIM}, "
            f"got {embedding_dim}"
        )

    # Preserve official dataset split.
    train_indices = np.where(splits == "train")[0]
    test_indices = np.where(splits == "test")[0]

    # Reproduce the same stratified 80/20 split
    # used by the single-frame baseline.
    train_indices, val_indices = train_test_split(
        train_indices,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=labels[train_indices]
    )

    X_train = embeddings[train_indices]
    X_val = embeddings[val_indices]
    X_test = embeddings[test_indices]

    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    # Fit scaler on training frames only.
    # Each frame is one observation for feature scaling.
    scaler = StandardScaler()

    X_train_flat = X_train.reshape(-1, embedding_dim)

    scaler.fit(X_train_flat)

    def transform_sequences(X):

        original_shape = X.shape

        X_flat = X.reshape(-1, embedding_dim)

        X_scaled = scaler.transform(X_flat)

        return X_scaled.reshape(original_shape).astype(np.float32)

    X_train = transform_sequences(X_train)
    X_val = transform_sequences(X_val)
    X_test = transform_sequences(X_test)

    # Encode labels.
    label_encoder = LabelEncoder()

    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    y_test = label_encoder.transform(y_test)

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
        scaler,
        num_frames,
        embedding_dim
    )


# ============================================================
# PYTORCH DATA LOADERS
# ============================================================

def create_data_loaders(
    X_train,
    X_val,
    X_test,
    y_train,
    y_val,
    y_test
):

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )

    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )

    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


# ============================================================
# TRAINING
# ============================================================

def train_model(model, train_loader, val_loader):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0

    for epoch in range(EPOCHS):

        # -------------------------
        # Training phase
        # -------------------------

        model.train()

        train_loss = 0.0

        for inputs, targets in train_loader:

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(
                outputs,
                targets
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------------------------
        # Validation phase
        # -------------------------

        model.eval()

        val_loss = 0.0

        with torch.no_grad():

            for inputs, targets in val_loader:

                inputs = inputs.to(DEVICE)
                targets = targets.to(DEVICE)

                outputs = model(inputs)

                loss = criterion(
                    outputs,
                    targets
                )

                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"Epoch [{epoch + 1:02d}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {val_loss:.4f}"
        )

        # -------------------------
        # Early stopping
        # -------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("\nEarly stopping triggered.")
            break

    if best_model_state is None:
        raise RuntimeError(
            "Training did not produce a valid model checkpoint."
        )

    model.load_state_dict(best_model_state)

    return model


# ============================================================
# EVALUATION
# ============================================================

def evaluate_model(model, test_loader, label_encoder):

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for inputs, targets in test_loader:

            inputs = inputs.to(DEVICE)

            outputs = model(inputs)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_targets.extend(
                targets.numpy()
            )

    accuracy = accuracy_score(
        all_targets,
        all_predictions
    )

    macro_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro"
    )

    print("\n========== TEST RESULTS ==========")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            all_targets,
            all_predictions,
            labels=np.arange(len(label_encoder.classes_)),
            target_names=label_encoder.classes_,
            zero_division=0
        )
    )

    matrix = confusion_matrix(
        all_targets,
        all_predictions,
        labels=np.arange(len(label_encoder.classes_))
    )

    print("Confusion Matrix:")
    print(matrix)

    return accuracy, macro_f1, matrix


# ============================================================
# SAVE MODEL AND RESULTS
# ============================================================

def save_results(
    model,
    accuracy,
    macro_f1,
    matrix,
    label_encoder,
    scaler,
    num_frames,
    embedding_dim
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save model and preprocessing parameters.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": label_encoder.classes_.tolist(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "input_dim": embedding_dim,
            "hidden_dim": HIDDEN_DIM,
            "num_layers": NUM_LAYERS,
            "num_frames": num_frames,
            "model_type": "temporal_gru",
            "random_state": RANDOM_STATE
        },
        RESULTS_DIR / "gru_temporal.pth"
    )

    # Save experiment metrics.
    metrics = {
        "model": "temporal_gru",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "test_samples": 180,
        "frames_per_video": int(num_frames),
        "embedding_dimension": int(embedding_dim),
        "hidden_dimension": HIDDEN_DIM,
        "num_layers": NUM_LAYERS
    }

    with (
        RESULTS_DIR / "gru_temporal_metrics.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Generate confusion matrix.
    plt.figure(figsize=(8, 6))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_encoder.classes_,
        yticklabels=label_encoder.classes_
    )

    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Temporal GRU Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "gru_temporal_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\n========== SAVED ARTIFACTS ==========")

    print(RESULTS_DIR / "gru_temporal.pth")
    print(RESULTS_DIR / "gru_temporal_metrics.json")
    print(PLOTS_DIR / "gru_temporal_confusion_matrix.png")


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(RANDOM_STATE)

    print(f"Using device: {DEVICE}")

    print("\nLoading temporal embeddings...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
        scaler,
        num_frames,
        embedding_dim
    ) = load_data()

    print("\n========== DATASET SUMMARY ==========")

    print(f"Training videos: {len(X_train)}")
    print(f"Validation videos: {len(X_val)}")
    print(f"Testing videos: {len(X_test)}")

    print(f"Frames per video: {num_frames}")
    print(f"Embedding dimension: {embedding_dim}")

    print(f"Classes: {list(label_encoder.classes_)}")

    print(
        f"Training tensor shape: {X_train.shape}"
    )

    train_loader, val_loader, test_loader = create_data_loaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    model = TemporalGRU(
        input_dim=embedding_dim,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=len(label_encoder.classes_),
        dropout=DROPOUT
    ).to(DEVICE)

    print("\n========== MODEL SUMMARY ==========")

    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(f"Trainable parameters: {total_parameters:,}")

    print("\nTraining temporal GRU...")

    model = train_model(
        model,
        train_loader,
        val_loader
    )

    accuracy, macro_f1, matrix = evaluate_model(
        model,
        test_loader,
        label_encoder
    )

    save_results(
        model,
        accuracy,
        macro_f1,
        matrix,
        label_encoder,
        scaler,
        num_frames,
        embedding_dim
    )


if __name__ == "__main__":
    main()
