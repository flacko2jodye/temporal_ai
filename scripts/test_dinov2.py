import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel


MODEL_NAME = "facebook/dinov2-small"


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    print("Loading image processor...")
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    print("Loading DINOv2-small...")
    model = AutoModel.from_pretrained(MODEL_NAME)

    model = model.to(device)
    model.eval()

    # Freeze all pretrained parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Create a test image
    image = Image.new("RGB", (224, 224), color=(120, 150, 180))

    inputs = processor(
        images=image,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    print("Running inference...")

    with torch.inference_mode():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state[:, 0, :]

    print("\n========== INFERENCE RESULTS ==========")
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding dimension: {embedding.shape[-1]}")
    print(f"Device used: {embedding.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB"
          if torch.cuda.is_available() else "GPU memory: N/A")

    assert embedding.shape == (1, 384), "Unexpected embedding shape!"

    print("\nDINOv2 inference test passed!")


if __name__ == "__main__":
    main()