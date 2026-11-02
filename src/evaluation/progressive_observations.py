
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_PATH = (
    PROJECT_ROOT / "data" / "processed" / "dinov2_hmdb51_split1.npz"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42

OBSERVATION_INDICES = {
    1: [4],
    2: [0, 7],
    4: [0, 2, 5, 7],
    8: list(range(8)),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Reproducibility
# --------------------------------------------------

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------
# Model
# --------------------------------------------------

class GRUTemporalClassifier(nn.Module):

    def __init__(
        self,
        input_dim=384,
        hidden_dim=128,
        num_classes=6,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        _, hidden = self.gru(x)

        final_hidden = hidden[-1]

        return self.classifier(final_hidden)


# --------------------------------------------------
# Observation selection
# --------------------------------------------------

def select_observations(X, count):
    """
    Select observations distributed across the complete video.

    X shape:
        (number_of_videos, number_of_frames, embedding_dim)
    """

    if count not in OBSERVATION_INDICES:
        raise ValueError(f"Unsupported observation count: {count}")

    indices = OBSERVATION_INDICES[count]

    if max(indices) >= X.shape[1]:
        raise ValueError(
            f"Selected frame index exceeds available frames. "
            f"Available: {X.shape[1]}, selected: {indices}"
        )

    return X[:, indices, :], indices


# --------------------------------------------------
# Training and evaluation
# --------------------------------------------------

def train_model(
    X_train,
    y_train,
    X_val,
    y_val,
    input_dim,
    num_classes,
):
    model = GRUTemporalClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=num_classes,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )

    val_X = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    val_y = torch.tensor(y_val, dtype=torch.long).to(DEVICE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_state = None
    patience = 8
    patience_counter = 0
    max_epochs = 50

    for epoch in range(max_epochs):

        model.train()
        total_train_loss = 0.0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        model.eval()

        with torch.no_grad():
            val_outputs = model(val_X)
            val_loss = criterion(val_outputs, val_y).item()

        average_train_loss = total_train_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1:02d}/{max_epochs} | "
            f"Train Loss: {average_train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def evaluate_model(model, X_test, y_test):

    model.eval()

    test_X = torch.tensor(
        X_test,
        dtype=torch.float32,
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(test_X)
        predictions = outputs.argmax(dim=1).cpu().numpy()

    accuracy = float(np.mean(predictions == y_test))

    # Macro F1 without requiring additional dependencies.
    num_classes = len(np.unique(y_test))
    f1_scores = []

    for class_id in range(num_classes):

        true_positive = np.sum(
            (predictions == class_id) & (y_test == class_id)
        )

        false_positive = np.sum(
            (predictions == class_id) & (y_test != class_id)
        )

        false_negative = np.sum(
            (predictions != class_id) & (y_test == class_id)
        )

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive > 0
            else 0.0
        )

        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative > 0
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall > 0
            else 0.0
        )

        f1_scores.append(f1)

    macro_f1 = float(np.mean(f1_scores))

    return accuracy, macro_f1


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    set_seed()

    print(f"Device: {DEVICE}")
    print("Loading DINOv2 embeddings...")

    data = np.load(EMBEDDING_PATH, allow_pickle=True)

    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"]
    splits = data["splits"]

    print(f"Embedding shape: {embeddings.shape}")

    if embeddings.shape[1] != 8:
        raise ValueError(
            "This experiment expects exactly 8 sampled frames per video."
        )

    # Encode labels consistently.
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    encoded_labels = label_encoder.transform(labels)

    train_indices = np.where(splits == "train")[0]
    test_indices = np.where(splits == "test")[0]

    X_train_full = embeddings[train_indices]
    y_train_full = encoded_labels[train_indices]

    X_test_full = embeddings[test_indices]
    y_test = encoded_labels[test_indices]

    # Same train/validation split for every observation condition.
    train_idx, val_idx = train_test_split(
        np.arange(len(y_train_full)),
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_train_full,
    )

    print(f"Training videos: {len(train_idx)}")
    print(f"Validation videos: {len(val_idx)}")
    print(f"Testing videos: {len(y_test)}")

    results = []

    # --------------------------------------------------
    # Run each observation condition
    # --------------------------------------------------

    for observation_count, selected_indices in OBSERVATION_INDICES.items():

        print("\n" + "=" * 60)
        print(f"EXPERIMENT: {observation_count} OBSERVATION(S)")
        print(f"Selected frame indices: {selected_indices}")
        print("=" * 60)

        set_seed()

        X_train_selected, actual_indices = select_observations(
            X_train_full,
            observation_count,
        )

        X_test_selected, _ = select_observations(
            X_test_full,
            observation_count,
        )

        # Split the selected training data into train and validation.
        X_train = X_train_selected[train_idx]
        X_val = X_train_selected[val_idx]

        y_train = y_train_full[train_idx]
        y_val = y_train_full[val_idx]

        # Fit scaler using training data only.
        scaler = StandardScaler()

        num_train, num_frames, embedding_dim = X_train.shape

        X_train_2d = X_train.reshape(-1, embedding_dim)
        X_val_2d = X_val.reshape(-1, embedding_dim)
        X_test_2d = X_test_selected.reshape(-1, embedding_dim)

        scaler.fit(X_train_2d)

        X_train_scaled = scaler.transform(
            X_train_2d
        ).reshape(num_train, num_frames, embedding_dim)

        X_val_scaled = scaler.transform(
            X_val_2d
        ).reshape(len(X_val), num_frames, embedding_dim)

        X_test_scaled = scaler.transform(
            X_test_2d
        ).reshape(len(X_test_selected), num_frames, embedding_dim)

        # Train model.
        model = train_model(
            X_train_scaled,
            y_train,
            X_val_scaled,
            y_val,
            embedding_dim,
            len(label_encoder.classes_),
        )

        # Evaluate on official test split.
        accuracy, macro_f1 = evaluate_model(
            model,
            X_test_scaled,
            y_test,
        )

        print(f"\nTest Accuracy: {accuracy:.4f}")
        print(f"Test Macro F1: {macro_f1:.4f}")

        # Save model and preprocessing information.
        checkpoint_path = (
            RESULTS_DIR
            / f"progressive_coverage_gru_{observation_count}_frames.pth"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "scaler": scaler,
                "classes": label_encoder.classes_,
                "observation_count": observation_count,
                "selected_frame_indices": actual_indices,
                "input_dim": embedding_dim,
                "hidden_dim": 128,
                "num_classes": len(label_encoder.classes_),
                "seed": RANDOM_SEED,
            },
            checkpoint_path,
        )

        results.append(
            {
                "observation_count": observation_count,
                "selected_frame_indices": actual_indices,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
            }
        )

    # --------------------------------------------------
    # Save experiment results
    # --------------------------------------------------

    output_path = (
        RESULTS_DIR / "progressive_observations_coverage_metrics.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "experiment": "coverage_controlled_progressive_observations",
                "dataset": "HMDB51 selected six classes",
                "embedding_model": "facebook/dinov2-small",
                "frames_per_video": 8,
                "observation_selection": OBSERVATION_INDICES,
                "random_seed": RANDOM_SEED,
                "results": results,
            },
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("FINAL PROGRESSIVE OBSERVATION RESULTS")
    print("=" * 60)

    for result in results:
        print(
            f"{result['observation_count']} frame(s) "
            f"{result['selected_frame_indices']}: "
            f"Accuracy={result['accuracy']:.4f}, "
            f"Macro F1={result['macro_f1']:.4f}"
        )

    print(f"\nResults saved to: {output_path}")
    print("Experiment completed successfully.")


if __name__ == "__main__":
    main()
