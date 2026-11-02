from pathlib import Path

import numpy as np
import torch

from src.models.gru_temporal import TemporalGRU


def load_temporal_gru(checkpoint_path, device):
    """
    Load a trained TemporalGRU checkpoint and its preprocessing metadata.

    Returns:
        model
        scaler_mean
        scaler_scale
        classes
        checkpoint
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = TemporalGRU(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(checkpoint["classes"]),
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model.to(device)
    model.eval()

    scaler_mean = checkpoint["scaler_mean"].astype(np.float32)
    scaler_scale = checkpoint["scaler_scale"].astype(np.float32)

    # Protect against division by zero.
    scaler_scale = np.where(
        scaler_scale == 0,
        1.0,
        scaler_scale,
    )

    classes = np.asarray(checkpoint["classes"])

    print("GRU checkpoint loaded.")
    print(f"Device: {device}")
    print(f"Classes: {classes.tolist()}")
    print(f"Frames per video: {checkpoint['num_frames']}")
    print(f"Embedding dimension: {checkpoint['input_dim']}")

    return model, scaler_mean, scaler_scale, classes, checkpoint