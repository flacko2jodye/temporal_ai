from pathlib import Path
import json

import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_DIR = PROJECT_ROOT / "results" / "class_confusion_analysis"
OUTPUT_DIR = PROJECT_ROOT / "results" / "model_error_comparison"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILES = {
    "single_frame": INPUT_DIR / "single_frame_predictions.csv",
    "mean_pool": INPUT_DIR / "mean_pool_predictions.csv",
    "gru": INPUT_DIR / "gru_predictions.csv",
}


# ---------------------------------------------------------
# Load and validate prediction files
# ---------------------------------------------------------

def load_predictions():
    predictions = {}

    for model_name, file_path in MODEL_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing prediction file: {file_path}\n"
                "Run scripts/analyze_class_confusions.py first."
            )

        df = pd.read_csv(file_path)

        required_columns = {
            "filename",
            "true_label",
            "predicted_label",
            "correct",
        }

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"{file_path.name} is missing columns: {sorted(missing)}"
            )

        if df["filename"].duplicated().any():
            raise ValueError(
                f"Duplicate filenames found in {file_path.name}"
            )

        predictions[model_name] = df[
            ["filename", "true_label", "predicted_label", "correct"]
        ].copy()

        predictions[model_name]["correct"] = (
            predictions[model_name]["correct"]
            .astype(str)
            .str.lower()
            .map({"true": True, "false": False, "1": True, "0": False})
        )

        if predictions[model_name]["correct"].isna().any():
            raise ValueError(
                f"Unexpected values in the correct column of "
                f"{file_path.name}"
            )

    return predictions


# ---------------------------------------------------------
# Merge predictions using filenames
# ---------------------------------------------------------

def merge_predictions(predictions):
    model_names = list(predictions.keys())

    merged = predictions[model_names[0]].rename(
        columns={
            "true_label": "true_label",
            "predicted_label": f"{model_names[0]}_prediction",
            "correct": f"{model_names[0]}_correct",
        }
    )

    for model_name in model_names[1:]:
        current = predictions[model_name].rename(
            columns={
                "true_label": f"{model_name}_true_label",
                "predicted_label": f"{model_name}_prediction",
                "correct": f"{model_name}_correct",
            }
        )

        merged = merged.merge(
            current,
            on="filename",
            how="outer",
            validate="one_to_one",
            indicator=f"_merge_{model_name}",
        )

        merge_column = f"_merge_{model_name}"

        if not (merged[merge_column] == "both").all():
            mismatches = merged.loc[
                merged[merge_column] != "both", "filename"
            ].tolist()

            raise ValueError(
                f"Prediction files do not contain identical video sets. "
                f"Mismatch examples: {mismatches[:10]}"
            )

        merged.drop(columns=[merge_column], inplace=True)

        true_label_column = f"{model_name}_true_label"

        if not (
            merged["true_label"] == merged[true_label_column]
        ).all():
            raise ValueError(
                f"Ground-truth labels differ for model {model_name}."
            )

        merged.drop(columns=[true_label_column], inplace=True)

    merged.sort_values("filename", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    return merged


# ---------------------------------------------------------
# Compare model performance on individual videos
# ---------------------------------------------------------

def summarize_errors(merged):
    models = ["single_frame", "mean_pool", "gru"]

    merged["all_models_correct"] = merged[
        [f"{model}_correct" for model in models]
    ].all(axis=1)

    merged["all_models_wrong"] = ~merged[
        [f"{model}_correct" for model in models]
    ].any(axis=1)

    merged["error_pattern"] = merged.apply(
        lambda row: " | ".join(
            model
            for model in models
            if not row[f"{model}_correct"]
        )
        or "none",
        axis=1,
    )

    total = len(merged)

    print("\n" + "=" * 72)
    print("OVERALL RESULTS")
    print("=" * 72)

    overall = {}

    for model in models:
        correct_count = int(merged[f"{model}_correct"].sum())
        incorrect_count = total - correct_count

        overall[model] = {
            "correct": correct_count,
            "incorrect": incorrect_count,
            "accuracy": correct_count / total,
        }

        print(
            f"{model:15s} "
            f"Correct: {correct_count:3d}/{total} | "
            f"Incorrect: {incorrect_count:3d} | "
            f"Accuracy: {correct_count / total:.4f}"
        )

    all_correct = int(merged["all_models_correct"].sum())
    all_wrong = int(merged["all_models_wrong"].sum())

    print(f"\nVideos all three models classify correctly: {all_correct}")
    print(f"Videos all three models misclassify:        {all_wrong}")

    # -----------------------------------------------------
    # Pairwise comparisons
    # -----------------------------------------------------

    pairwise = {}

    pairs = [
        ("single_frame", "mean_pool"),
        ("single_frame", "gru"),
        ("mean_pool", "gru"),
    ]

    print("\n" + "=" * 72)
    print("PAIRWISE ERROR COMPARISON")
    print("=" * 72)

    for model_a, model_b in pairs:
        a_correct = merged[f"{model_a}_correct"]
        b_correct = merged[f"{model_b}_correct"]

        a_only_correct = int((a_correct & ~b_correct).sum())
        b_only_correct = int((~a_correct & b_correct).sum())
        both_correct = int((a_correct & b_correct).sum())
        both_wrong = int((~a_correct & ~b_correct).sum())

        pair_key = f"{model_a}_vs_{model_b}"

        pairwise[pair_key] = {
            f"{model_a}_correct_{model_b}_wrong": a_only_correct,
            f"{model_b}_correct_{model_a}_wrong": b_only_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
        }

        print(f"\n{model_a} vs {model_b}")
        print(f"  Both correct:                  {both_correct}")
        print(f"  Both wrong:                    {both_wrong}")
        print(f"  {model_a} correct, {model_b} wrong: {a_only_correct}")
        print(f"  {model_b} correct, {model_a} wrong: {b_only_correct}")

    # -----------------------------------------------------
    # Class-level error comparison
    # -----------------------------------------------------

    class_rows = []

    for class_name, group in merged.groupby("true_label"):
        row = {
            "class": class_name,
            "support": len(group),
        }

        for model in models:
            row[f"{model}_correct"] = int(
                group[f"{model}_correct"].sum()
            )
            row[f"{model}_accuracy"] = float(
                group[f"{model}_correct"].mean()
            )
            row[f"{model}_errors"] = int(
                (~group[f"{model}_correct"]).sum()
            )

        class_rows.append(row)

    class_summary = pd.DataFrame(class_rows).sort_values("class")

    print("\n" + "=" * 72)
    print("PER-CLASS COMPARISON")
    print("=" * 72)

    display_columns = [
        "class",
        "support",
        "single_frame_accuracy",
        "mean_pool_accuracy",
        "gru_accuracy",
        "single_frame_errors",
        "mean_pool_errors",
        "gru_errors",
    ]

    print(
        class_summary[display_columns]
        .round(4)
        .to_string(index=False)
    )

    # -----------------------------------------------------
    # Specific videos where the GRU and mean pooling differ
    # -----------------------------------------------------

    mean_correct = merged["mean_pool_correct"]
    gru_correct = merged["gru_correct"]

    gru_fixed = merged[~mean_correct & gru_correct].copy()
    gru_regressed = merged[mean_correct & ~gru_correct].copy()

    print("\n" + "=" * 72)
    print("GRU VS MEAN POOLING")
    print("=" * 72)

    print(
        "GRU corrects mean-pooling errors: "
        f"{len(gru_fixed)} videos"
    )

    print(
        "GRU introduces errors on mean-pooling successes: "
        f"{len(gru_regressed)} videos"
    )

    print("\nVideos corrected by the GRU:")
    if gru_fixed.empty:
        print("  None")
    else:
        print(
            gru_fixed[
                [
                    "filename",
                    "true_label",
                    "mean_pool_prediction",
                    "gru_prediction",
                ]
            ].to_string(index=False)
        )

    print("\nVideos where the GRU regresses:")
    if gru_regressed.empty:
        print("  None")
    else:
        print(
            gru_regressed[
                [
                    "filename",
                    "true_label",
                    "mean_pool_prediction",
                    "gru_prediction",
                ]
            ].to_string(index=False)
        )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    merged.to_csv(
        OUTPUT_DIR / "all_model_predictions.csv",
        index=False,
    )

    class_summary.to_csv(
        OUTPUT_DIR / "per_class_comparison.csv",
        index=False,
    )

    gru_fixed.to_csv(
        OUTPUT_DIR / "gru_corrected_errors.csv",
        index=False,
    )

    gru_regressed.to_csv(
        OUTPUT_DIR / "gru_regressed_errors.csv",
        index=False,
    )

    summary = {
        "total_videos": total,
        "overall": overall,
        "all_models_correct": all_correct,
        "all_models_wrong": all_wrong,
        "pairwise_comparisons": pairwise,
        "gru_corrected_mean_pool_errors": len(gru_fixed),
        "gru_regressed_on_mean_pool_successes": len(gru_regressed),
    }

    with open(
        OUTPUT_DIR / "error_comparison_summary.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 72)
    print("SAVED ARTIFACTS")
    print("=" * 72)
    print(f"Output directory: {OUTPUT_DIR}")

    for file_path in sorted(OUTPUT_DIR.iterdir()):
        print(f"  {file_path.name}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    predictions = load_predictions()
    merged = merge_predictions(predictions)
    summarize_errors(merged)


if __name__ == "__main__":
    main()
    