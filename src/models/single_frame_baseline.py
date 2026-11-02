
import json
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

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL
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

            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================

def load_data():

    if not EMBEDDING_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_FILE}"
        )

    with np.load(EMBEDDING_FILE, allow_pickle=False) as data:
        embeddings = data["embeddings"]
        labels = data["labels"]
        splits = data["splits"]

    # Use only the middle frame of each video.
    middle_frame_index = embeddings.shape[1] // 2

    X = embeddings[:, middle_frame_index, :]
    y = labels

    # Preserve the official train/test split.
    train_indices = np.where(splits == "train")[0]
    test_indices = np.where(splits == "test")[0]

    X_train_full = X[train_indices]
    y_train_full = y[train_indices]

    X_test = X[test_indices]
    y_test = y[test_indices]

    # Create a stratified validation split from training data.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train_full
    )

    # Fit preprocessing only on training data.
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Encode class labels.
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
        scaler
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
            loss = criterion(outputs, targets)

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
                loss = criterion(outputs, targets)

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
# SAVE MODEL AND EXPERIMENT RESULTS
# ============================================================

def save_results(
    model,
    accuracy,
    macro_f1,
    matrix,
    label_encoder,
    scaler
):

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save trained model and preprocessing parameters.
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": label_encoder.classes_.tolist(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "input_dim": 384,
            "model_type": "single_frame_mlp",
            "random_state": RANDOM_STATE
        },
        RESULTS_DIR / "single_frame_baseline.pth"
    )

    # Save metrics.
    metrics = {
        "model": "single_frame_mlp",
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "test_samples": 180,
        "frames_per_video": 1,
        "embedding_dimension": 384
    }

    with (
        RESULTS_DIR / "single_frame_metrics.json"
    ).open("w", encoding="utf-8") as file:

        json.dump(
            metrics,
            file,
            indent=4
        )

    # Generate confusion matrix visualization.
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
    plt.title("Single-Frame Baseline Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        PLOTS_DIR / "single_frame_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    print("\n========== SAVED ARTIFACTS ==========")
    print(RESULTS_DIR / "single_frame_baseline.pth")
    print(RESULTS_DIR / "single_frame_metrics.json")
    print(PLOTS_DIR / "single_frame_confusion_matrix.png")


# ============================================================
# MAIN
# ============================================================

def main():

    print(f"Using device: {DEVICE}")

    print("\nLoading embeddings...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        label_encoder,
        scaler
    ) = load_data()

    print("\n========== DATASET SUMMARY ==========")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Input dimensions: {X_train.shape[1]}")
    print(f"Classes: {list(label_encoder.classes_)}")

    train_loader, val_loader, test_loader = create_data_loaders(
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test
    )

    model = SingleFrameClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_)
    ).to(DEVICE)

    print("\nTraining single-frame baseline...")

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
        scaler
    )


if __name__ == "__main__":
    main()
