from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = [
    "results/single_frame_baseline.pth",
    "results/mean_pool_baseline.pth",
    "results/gru_temporal.pth",
]


def inspect_checkpoint(relative_path):
    path = PROJECT_ROOT / relative_path

    print("\n" + "=" * 70)
    print(f"CHECKPOINT: {relative_path}")
    print("=" * 70)

    if not path.exists():
        print("File not found.")
        return

    checkpoint = torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )

    print(f"Top-level object: {type(checkpoint).__name__}")

    if not isinstance(checkpoint, dict):
        print("Checkpoint is not a dictionary.")
        return

    print("\nTop-level keys:")
    for key, value in checkpoint.items():
        if isinstance(value, dict):
            print(f"  {key}: dictionary ({len(value)} entries)")
        elif hasattr(value, "shape"):
            print(f"  {key}: shape={value.shape}")
        else:
            print(f"  {key}: {type(value).__name__}")

    print("\nModel state dictionary:")
    state_dict = checkpoint.get("model_state_dict")

    if state_dict is None:
        state_dict = checkpoint.get("state_dict")

    if isinstance(state_dict, dict):
        for key, value in state_dict.items():
            if hasattr(value, "shape"):
                print(f"  {key}: {tuple(value.shape)}")
            else:
                print(f"  {key}: {type(value).__name__}")
    else:
        print("  No model_state_dict or state_dict key found.")

    print("\nAdditional metadata:")
    for key in [
        "classes",
        "class_names",
        "input_dim",
        "hidden_dim",
        "num_layers",
        "scaler",
        "epoch",
        "best_val_loss",
    ]:
        if key in checkpoint:
            value = checkpoint[key]

            if key == "scaler":
                print(f"  {key}: {type(value).__name__}")
            else:
                print(f"  {key}: {value}")


def main():
    print("PyTorch:", torch.__version__)

    for checkpoint in CHECKPOINTS:
        inspect_checkpoint(checkpoint)


if __name__ == "__main__":
    main()