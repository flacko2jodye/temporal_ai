
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUTPUT = RESULTS / "final_summary"

OUTPUT.mkdir(parents=True, exist_ok=True)


def load_json(relative_path):
    path = RESULTS / relative_path
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# 1. Main model comparison
# --------------------------------------------------

single = load_json("single_frame_metrics.json")
mean_pool = load_json("mean_pool_metrics.json")
gru = load_json("gru_temporal_metrics.json")
clip = load_json("clip_zero_shot/metrics.json")

models = [
    {
        "model": "DINOv2 Single-frame MLP",
        "input": "1 frame",
        "accuracy": single["accuracy"],
        "macro_f1": single["macro_f1"],
    },
    {
        "model": "DINOv2 Mean-pooling MLP",
        "input": "8 frames",
        "accuracy": mean_pool["accuracy"],
        "macro_f1": mean_pool["macro_f1"],
    },
    {
        "model": "DINOv2 Temporal GRU",
        "input": "8 ordered frames",
        "accuracy": gru["accuracy"],
        "macro_f1": gru["macro_f1"],
    },
    {
        "model": "CLIP Zero-shot",
        "input": "8-frame score averaging",
        "accuracy": clip["accuracy"],
        "macro_f1": clip["macro_f1"],
    },
]

model_df = pd.DataFrame(models)
model_df.to_csv(OUTPUT / "model_comparison.csv", index=False)


# --------------------------------------------------
# 2. Progressive observations
# --------------------------------------------------

progressive = load_json(
    "progressive_observations_coverage_metrics.json"
)

progressive_df = pd.DataFrame(progressive["results"])
progressive_df.to_csv(
    OUTPUT / "progressive_observations.csv",
    index=False,
)


# --------------------------------------------------
# 3. Temporal order sensitivity
# --------------------------------------------------

order = load_json("temporal_order_metrics.json")
conditions = order["conditions"]

order_rows = []

for name, metrics in conditions.items():
    order_rows.append({
        "condition": name,
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
    })

order_df = pd.DataFrame(order_rows)
order_df.to_csv(
    OUTPUT / "temporal_order_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 4. Frame embedding ablation
# --------------------------------------------------

ablation = load_json(
    "frame_ablation/frame_ablation_metrics.json"
)

ablation_rows = []

for name, metrics in ablation["results"].items():
    ablation_rows.append({
        "condition": name,
        "frame_removed": metrics["frame_removed"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "accuracy_change": metrics.get("accuracy_change", 0.0),
        "macro_f1_change": metrics.get("macro_f1_change", 0.0),
    })

ablation_df = pd.DataFrame(ablation_rows)
ablation_df.to_csv(
    OUTPUT / "frame_ablation_comparison.csv",
    index=False,
)


# --------------------------------------------------
# 5. Main model comparison plot
# --------------------------------------------------

x = np.arange(len(model_df))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

ax.bar(
    x - width / 2,
    model_df["accuracy"] * 100,
    width,
    label="Accuracy",
)

ax.bar(
    x + width / 2,
    model_df["macro_f1"] * 100,
    width,
    label="Macro-F1",
)

ax.set_ylabel("Score (%)")
ax.set_title("Model Performance Comparison")
ax.set_xticks(x)
ax.set_xticklabels(model_df["model"], rotation=15, ha="right")
ax.set_ylim(0, 100)
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT / "model_comparison.png",
    dpi=300,
)
plt.close(fig)


# --------------------------------------------------
# 6. Progressive observations plot
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    progressive_df["observation_count"],
    progressive_df["accuracy"] * 100,
    marker="o",
    label="Accuracy",
)

ax.plot(
    progressive_df["observation_count"],
    progressive_df["macro_f1"] * 100,
    marker="s",
    label="Macro-F1",
)

ax.set_xlabel("Number of observed frames")
ax.set_ylabel("Score (%)")
ax.set_title("Performance Across Observation Counts")
ax.set_xticks([1, 2, 4, 8])
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT / "progressive_observations.png",
    dpi=300,
)
plt.close(fig)


# --------------------------------------------------
# 7. Temporal order plot
# --------------------------------------------------

plot_order = order_df.copy()
plot_order["condition"] = plot_order["condition"].replace({
    "original": "Chronological",
    "reversed": "Reversed",
})

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(
    plot_order["condition"],
    plot_order["accuracy"] * 100,
)

ax.set_ylabel("Accuracy (%)")
ax.set_title("Temporal Order Sensitivity")
ax.tick_params(axis="x", rotation=35)
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT / "temporal_order_sensitivity.png",
    dpi=300,
)
plt.close(fig)


# --------------------------------------------------
# 8. Frame ablation plot
# --------------------------------------------------

frame_rows = ablation_df[
    ablation_df["frame_removed"].notna()
].copy()

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(
    frame_rows["frame_removed"].astype(int).astype(str),
    frame_rows["accuracy"] * 100,
)

ax.axhline(
    ablation["results"]["original"]["accuracy"] * 100,
    linestyle="--",
    label="Original GRU",
)

ax.set_xlabel("Frame embedding removed")
ax.set_ylabel("Accuracy (%)")
ax.set_title("Individual Frame Embedding Ablation")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
fig.savefig(
    OUTPUT / "frame_ablation.png",
    dpi=300,
)
plt.close(fig)


# --------------------------------------------------
# 9. Machine-readable summary
# --------------------------------------------------

summary = {
    "dataset": "HMDB51 selected six classes",
    "test_samples": 180,
    "classes": [
        "brush_hair",
        "clap",
        "drink",
        "eat",
        "pour",
        "wave",
    ],
    "models": models,
    "temporal_order": order["shuffle_statistics"],
    "progressive_observations": progressive["results"],
    "frame_ablation": ablation["results"],
}

with (OUTPUT / "final_results_summary.json").open(
    "w",
    encoding="utf-8",
) as file:
    json.dump(summary, file, indent=2)


print("\nFinal results summary generated successfully.")
print(f"Output directory: {OUTPUT}")

print("\nMain model comparison:")
print(model_df.to_string(index=False))

print("\nTemporal shuffle statistics:")
print(json.dumps(order["shuffle_statistics"], indent=2))

print("\nGenerated files:")
for path in sorted(OUTPUT.iterdir()):
    print(f" - {path.name}")
