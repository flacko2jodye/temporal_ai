
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

EMBEDDING_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dinov2_hmdb51_split1.npz"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "temporal_analysis"
PLOT_DIR = OUTPUT_DIR / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_SEED = 42
SEQUENCES_PER_CLASS = 2


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def normalize_vectors(vectors):
    """L2-normalize vectors safely."""

    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)

    return vectors / np.maximum(norms, 1e-12)


def calculate_adjacent_cosine(embeddings):
    """
    Calculate cosine similarity between consecutive
    frame embeddings.

    Input shape:
        (videos, frames, embedding_dimension)

    Output shape:
        (videos, frames - 1)
    """

    current_frames = embeddings[:, :-1, :]
    next_frames = embeddings[:, 1:, :]

    current_normalized = normalize_vectors(current_frames)
    next_normalized = normalize_vectors(next_frames)

    similarities = np.sum(
        current_normalized * next_normalized,
        axis=-1,
    )

    return similarities


def calculate_sequence_representations(embeddings):
    """
    Generate one representation per video by mean pooling
    its frame embeddings.

    Input:
        (videos, frames, embedding_dimension)

    Output:
        (videos, embedding_dimension)
    """

    pooled = np.mean(embeddings, axis=1)

    return normalize_vectors(pooled)


def calculate_class_similarity(sequence_vectors, labels, classes):
    """
    Calculate:
    - Mean pairwise within-class cosine similarity.
    - Mean pairwise between-class cosine similarity.
    - Class centroid cosine similarity matrix.
    """

    similarity_matrix = cosine_similarity(sequence_vectors)

    intra_class_scores = []
    inter_class_scores = []
    class_centroids = []

    for class_name in classes:

        class_indices = np.where(labels == class_name)[0]

        class_similarities = similarity_matrix[
            np.ix_(class_indices, class_indices)
        ]

        upper_indices = np.triu_indices(
            len(class_indices),
            k=1,
        )

        intra_scores = class_similarities[upper_indices]

        intra_class_scores.extend(intra_scores.tolist())

        centroid = np.mean(
            sequence_vectors[class_indices],
            axis=0,
        )

        centroid = normalize_vectors(
            centroid.reshape(1, -1)
        )[0]

        class_centroids.append(centroid)

    for i in range(len(labels)):

        for j in range(i + 1, len(labels)):

            if labels[i] != labels[j]:
                inter_class_scores.append(
                    similarity_matrix[i, j]
                )

    centroid_matrix = cosine_similarity(
        np.asarray(class_centroids)
    )

    return (
        np.asarray(intra_class_scores),
        np.asarray(inter_class_scores),
        centroid_matrix,
    )


# --------------------------------------------------
# Plot 1: Adjacent-frame similarity
# --------------------------------------------------

def plot_adjacent_similarity(adjacent_similarity, labels, classes):

    plt.figure(figsize=(12, 6))

    data = [
        adjacent_similarity[labels == class_name].flatten()
        for class_name in classes
    ]

    plt.boxplot(
        data,
        tick_labels=classes,
        showmeans=True,
    )

    plt.title("Adjacent-Frame Cosine Similarity by Action")
    plt.xlabel("Action Class")
    plt.ylabel("Cosine Similarity")
    plt.xticks(rotation=25)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "adjacent_frame_similarity.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Plot 2: Temporal variation
# --------------------------------------------------

def plot_temporal_variation(temporal_variation, labels, classes):

    plt.figure(figsize=(10, 6))

    class_means = [
        np.mean(temporal_variation[labels == class_name])
        for class_name in classes
    ]

    class_stds = [
        np.std(temporal_variation[labels == class_name])
        for class_name in classes
    ]

    plt.bar(
        classes,
        class_means,
        yerr=class_stds,
        capsize=5,
    )

    plt.title("Mean Temporal Embedding Variation by Action")
    plt.xlabel("Action Class")
    plt.ylabel("Mean Cosine Distance Between Adjacent Frames")
    plt.xticks(rotation=25)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    output_path = PLOT_DIR / "temporal_variation_by_class.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Plot 3: PCA temporal trajectories
# --------------------------------------------------

def plot_embedding_trajectories(
    embeddings,
    labels,
    splits,
    filenames,
    classes,
):

    num_videos, num_frames, embedding_dim = embeddings.shape

    flattened_embeddings = embeddings.reshape(
        num_videos * num_frames,
        embedding_dim,
    )

    pca = PCA(n_components=2)

    projected = pca.fit_transform(flattened_embeddings)

    projected = projected.reshape(
        num_videos,
        num_frames,
        2,
    )

    # Use test videos for the visualization.
    test_indices = np.where(splits == "test")[0]

    selected_indices = []

    for class_name in classes:

        class_indices = [
            index
            for index in test_indices
            if labels[index] == class_name
        ]

        # Deterministic selection.
        selected_indices.extend(
            class_indices[:SEQUENCES_PER_CLASS]
        )

    plt.figure(figsize=(14, 10))

    for index in selected_indices:

        trajectory = projected[index]

        plt.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            marker="o",
            linewidth=1.5,
            markersize=4,
            label=f"{labels[index]}",
            alpha=0.8,
        )

        # Annotate the first and last observations.
        plt.annotate(
            "Start",
            (trajectory[0, 0], trajectory[0, 1]),
            fontsize=7,
        )

        plt.annotate(
            "End",
            (trajectory[-1, 0], trajectory[-1, 1]),
            fontsize=7,
        )

    plt.title("DINOv2 Embedding Trajectories Across Video Sequences")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid(alpha=0.3)

    # Avoid a large repeated legend for every video.
    handles, legend_labels = plt.gca().get_legend_handles_labels()

    unique_legend = {}

    for handle, legend_label in zip(handles, legend_labels):
        if legend_label not in unique_legend:
            unique_legend[legend_label] = handle

    plt.legend(
        unique_legend.values(),
        unique_legend.keys(),
        title="Action Class",
        loc="best",
    )

    plt.tight_layout()

    output_path = PLOT_DIR / "temporal_embedding_trajectories.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    explained_variance = pca.explained_variance_ratio_

    return {
        "explained_variance_pc1": float(explained_variance[0]),
        "explained_variance_pc2": float(explained_variance[1]),
        "explained_variance_total": float(
            np.sum(explained_variance)
        ),
        "visualized_sequences": len(selected_indices),
        "visualized_videos": [
            str(filenames[index])
            for index in selected_indices
        ],
    }


# --------------------------------------------------
# Plot 4: Class similarity heatmap
# --------------------------------------------------

def plot_class_similarity(centroid_matrix, classes):

    plt.figure(figsize=(9, 7))

    image = plt.imshow(
        centroid_matrix,
        interpolation="nearest",
        aspect="auto",
        vmin=-1,
        vmax=1,
    )

    plt.colorbar(
        image,
        label="Cosine Similarity",
    )

    plt.xticks(
        range(len(classes)),
        classes,
        rotation=30,
        ha="right",
    )

    plt.yticks(
        range(len(classes)),
        classes,
    )

    plt.title("Cosine Similarity Between Action-Class Centroids")

    for row in range(len(classes)):
        for column in range(len(classes)):

            plt.text(
                column,
                row,
                f"{centroid_matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = PLOT_DIR / "class_similarity_heatmap.png"

    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


# --------------------------------------------------
# Main analysis
# --------------------------------------------------

def main():

    np.random.seed(RANDOM_SEED)

    print("=" * 60)
    print("TEMPORAL EMBEDDING ANALYSIS")
    print("=" * 60)

    if not EMBEDDING_PATH.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {EMBEDDING_PATH}"
        )

    print("\nLoading embeddings...")

    data = np.load(
        EMBEDDING_PATH,
        allow_pickle=True,
    )

    embeddings = data["embeddings"].astype(np.float32)
    labels = data["labels"].astype(str)
    splits = data["splits"].astype(str)
    filenames = data["filenames"].astype(str)

    num_videos, num_frames, embedding_dim = embeddings.shape

    classes = sorted(np.unique(labels).tolist())

    print(f"Videos: {num_videos}")
    print(f"Frames per video: {num_frames}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Classes: {classes}")

    if not np.isfinite(embeddings).all():
        raise ValueError("Embeddings contain NaN or infinite values.")

    # --------------------------------------------------
    # Analysis 1: Adjacent-frame cosine similarity
    # --------------------------------------------------

    print("\n[1/4] Calculating adjacent-frame similarity...")

    adjacent_similarity = calculate_adjacent_cosine(
        embeddings
    )

    temporal_variation = 1.0 - np.mean(
        adjacent_similarity,
        axis=1,
    )

    plot_adjacent_similarity(
        adjacent_similarity,
        labels,
        classes,
    )

    plot_temporal_variation(
        temporal_variation,
        labels,
        classes,
    )

    # --------------------------------------------------
    # Analysis 2: Per-video temporal statistics
    # --------------------------------------------------

    print("\n[2/4] Generating per-video temporal statistics...")

    video_records = []

    for index in range(num_videos):

        video_records.append(
            {
                "filename": filenames[index],
                "label": labels[index],
                "split": splits[index],
                "mean_adjacent_cosine": float(
                    np.mean(adjacent_similarity[index])
                ),
                "std_adjacent_cosine": float(
                    np.std(adjacent_similarity[index])
                ),
                "mean_temporal_variation": float(
                    temporal_variation[index]
                ),
            }
        )

    video_df = pd.DataFrame(video_records)

    video_csv_path = OUTPUT_DIR / "per_video_temporal_statistics.csv"

    video_df.to_csv(
        video_csv_path,
        index=False,
    )

    print(f"Saved: {video_csv_path}")

    # Class-level summary.
    class_records = []

    for class_name in classes:

        class_values = temporal_variation[
            labels == class_name
        ]

        class_records.append(
            {
                "class": class_name,
                "video_count": len(class_values),
                "mean_temporal_variation": float(
                    np.mean(class_values)
                ),
                "std_temporal_variation": float(
                    np.std(class_values)
                ),
                "min_temporal_variation": float(
                    np.min(class_values)
                ),
                "max_temporal_variation": float(
                    np.max(class_values)
                ),
            }
        )

    class_df = pd.DataFrame(class_records)

    class_csv_path = OUTPUT_DIR / "class_temporal_statistics.csv"

    class_df.to_csv(
        class_csv_path,
        index=False,
    )

    print(f"Saved: {class_csv_path}")

    # --------------------------------------------------
    # Analysis 3: Embedding trajectories
    # --------------------------------------------------

    print("\n[3/4] Generating PCA embedding trajectories...")

    trajectory_results = plot_embedding_trajectories(
        embeddings,
        labels,
        splits,
        filenames,
        classes,
    )

    # --------------------------------------------------
    # Analysis 4: Inter-class similarity
    # --------------------------------------------------

    print("\n[4/4] Calculating inter-class similarity...")

    sequence_vectors = calculate_sequence_representations(
        embeddings
    )

    intra_scores, inter_scores, centroid_matrix = (
        calculate_class_similarity(
            sequence_vectors,
            labels,
            classes,
        )
    )

    plot_class_similarity(
        centroid_matrix,
        classes,
    )

    similarity_summary = {
        "mean_intra_class_similarity": float(
            np.mean(intra_scores)
        ),
        "std_intra_class_similarity": float(
            np.std(intra_scores)
        ),
        "mean_inter_class_similarity": float(
            np.mean(inter_scores)
        ),
        "std_inter_class_similarity": float(
            np.std(inter_scores)
        ),
    }

    centroid_df = pd.DataFrame(
        centroid_matrix,
        index=classes,
        columns=classes,
    )

    centroid_csv_path = OUTPUT_DIR / "class_centroid_similarity.csv"

    centroid_df.to_csv(centroid_csv_path)

    # --------------------------------------------------
    # Save consolidated results
    # --------------------------------------------------

    summary = {
        "dataset": "HMDB51 selected six classes",
        "embedding_model": "facebook/dinov2-small",
        "num_videos": int(num_videos),
        "num_frames_per_video": int(num_frames),
        "embedding_dimension": int(embedding_dim),
        "classes": classes,
        "adjacent_frame_similarity": {
            "overall_mean": float(
                np.mean(adjacent_similarity)
            ),
            "overall_std": float(
                np.std(adjacent_similarity)
            ),
            "per_class": {
                class_name: {
                    "mean": float(
                        np.mean(
                            adjacent_similarity[
                                labels == class_name
                            ]
                        )
                    ),
                    "std": float(
                        np.std(
                            adjacent_similarity[
                                labels == class_name
                            ]
                        )
                    ),
                }
                for class_name in classes
            },
        },
        "temporal_variation": {
            "overall_mean": float(
                np.mean(temporal_variation)
            ),
            "overall_std": float(
                np.std(temporal_variation)
            ),
        },
        "sequence_similarity": similarity_summary,
        "pca_trajectory": trajectory_results,
        "class_centroid_similarity": {
            class_name: {
                other_class: float(
                    centroid_matrix[i, j]
                )
                for j, other_class in enumerate(classes)
            }
            for i, class_name in enumerate(classes)
        },
    }

    summary_path = OUTPUT_DIR / "temporal_analysis_summary.json"

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 60)
    print("TEMPORAL ANALYSIS COMPLETED")
    print("=" * 60)

    print(
        "Overall adjacent-frame cosine similarity:",
        f"{np.mean(adjacent_similarity):.4f}",
    )

    print(
        "Overall temporal variation:",
        f"{np.mean(temporal_variation):.4f}",
    )

    print(
        "Mean intra-class similarity:",
        f"{np.mean(intra_scores):.4f}",
    )

    print(
        "Mean inter-class similarity:",
        f"{np.mean(inter_scores):.4f}",
    )

    print(
        "PCA variance explained:",
        f"{trajectory_results['explained_variance_total']:.4f}",
    )

    print(f"\nAll outputs saved under: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
