import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.embeddings.extract_embeddings import load_inventory


def load_test_inventory(expected_test_videos=180):
    """
    Load and validate the official readable test split.
    """

    inventory = load_inventory()

    test_rows = [
        row for row in inventory
        if row["split"] == "test"
        and row["readable"].lower() == "true"
    ]

    if len(test_rows) != expected_test_videos:
        raise ValueError(
            f"Expected {expected_test_videos} test videos, "
            f"found {len(test_rows)}."
        )

    print(f"Official test videos: {len(test_rows)}")

    return test_rows


def evaluate_gru_predictions(
    model,
    embeddings,
    labels,
    scaler_mean,
    scaler_scale,
    classes,
    device,
    num_frames,
    batch_size=32,
):
    """
    Scale embeddings using the training scaler and evaluate
    predictions using a frozen temporal GRU.
    """

    X = np.stack(embeddings).astype(np.float32)

    expected_shape = (num_frames, len(scaler_mean))

    if X.shape[1:] != expected_shape:
        raise ValueError(
            f"Unexpected embedding tensor shape: {X.shape}. "
            f"Expected (N, {expected_shape[0]}, "
            f"{expected_shape[1]})."
        )

    X = (
        X - scaler_mean.reshape(1, 1, -1)
    ) / scaler_scale.reshape(1, 1, -1)

    X = X.astype(np.float32)

    class_to_index = {
        name: index
        for index, name in enumerate(classes)
    }

    unknown = set(labels) - set(class_to_index)

    if unknown:
        raise ValueError(
            f"Unknown labels: {sorted(unknown)}"
        )

    y_true = np.asarray(
        [class_to_index[label] for label in labels],
        dtype=np.int64,
    )

    predictions = []

    with torch.inference_mode():

        for start in range(0, len(X), batch_size):

            batch = torch.from_numpy(
                X[start:start + batch_size]
            ).to(device)

            logits = model(batch)

            predictions.extend(
                torch.argmax(logits, dim=1)
                .cpu()
                .numpy()
                .tolist()
            )

    predictions = np.asarray(
        predictions,
        dtype=np.int64,
    )

    accuracy = accuracy_score(
        y_true,
        predictions,
    )

    macro_f1 = f1_score(
        y_true,
        predictions,
        average="macro",
        zero_division=0,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(classes))),
        target_names=classes.tolist(),
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(classes))),
    )

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "predictions": predictions,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }