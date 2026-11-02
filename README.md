# Temporal & Multimodal Visual Intelligence

**Research prototype | HMDB51 action recognition | DINOv2, MLP, GRU, and CLIP**

## 1. Research question and scope

I built this project to investigate a specific question from the take-home assessment:

> Can a model extract and use information from a sequence of visual observations that is unavailable from a single frame?

I distinguish **access to additional frames** from **learning their chronological relationships**. An order-invariant multi-frame baseline is essential: a recurrent model may appear better than a single-frame classifier simply because it receives more visual evidence.

The implemented study uses six HMDB51 action classes, frozen frame-level DINOv2 representations, three supervised classifiers, an independent zero-shot CLIP baseline, and controlled perturbation experiments. The objective is experimental insight rather than state-of-the-art accuracy.

**Main finding:** mean pooling of eight frame embeddings achieves **72.78% accuracy**, compared with **69.44%** for a single frame and **71.67%** for an ordered GRU. Multiple observations help in this setup; a reliable advantage from chronological order is **not demonstrated**.

- [Research report (PDF)](docs/research_report.pdf)
- [Research report (Markdown)](docs/research_report.md)
- [Implemented and proposed architecture](docs/architecture.md)

## 2. Dataset and official split

**Source:** [HMDB51 — Serre Lab](https://serre.lab.clps.brown.edu/hmdb51.html). I use the original unstabilized videos and **official split 1** assignments for:

`brush_hair`, `clap`, `drink`, `eat`, `pour`, and `wave`.

| Item | Count |
|---|---:|
| Videos inventoried in the six classes | 719 |
| Train videos (official split 1) | 420 |
| Test videos (official split 1) | 180 |
| Excluded from split 1 training/testing | 119 |
| Train/validation subsets used for fitting | 336 / 84 |
| Videos with saved DINOv2 embeddings | 600 |

There are **70 train and 30 test clips per class**. The dataset inventory recorded all 719 available videos as readable. The 119 videos not assigned to train/test in official split 1 were not used for classifier fitting or testing. The validation set is a stratified subset of the official training partition; the official test set is not used for model fitting or scaler estimation.

### Download and arrange the data

Download the original HMDB51 videos (`hmdb51_org.rar`) and official split archive (`test_train_splits.rar`) from the dataset provider or an authoritative mirror. These large archives are **not bundled** with the submission. Extract the six relevant action directories from the video archive, and place the official split text files at the paths shown below.

```text
project-root/
  data/
    raw/
      hmdb51/
        brush_hair/      # video files
        clap/
        drink/
        eat/
        pour/
        wave/
    splits/
      testTrainMulti_7030_splits/
        brush_hair_test_split1.txt
        clap_test_split1.txt
        drink_test_split1.txt
        eat_test_split1.txt
        pour_test_split1.txt
        wave_test_split1.txt
    processed/             # created by feature extraction
```

**Important:** the class folder and split filename spellings must match the scripts. Extracting `hmdb51_org.rar` can require extracting the per-class archives inside it. The final layout must contain the actual video files inside each class directory, not only nested `.rar` files.

No unverified Hugging Face CLI download command is required; the dataset can be obtained and placed manually. The archived source files, local frames, per-video embedding checkpoints, and consolidated embeddings remain local and are excluded by `.gitignore`.

### Dataset validation and inventory

From the project root:

```powershell
python scripts/check_dataset_integrity.py
python scripts/inspect_video_dataset.py
```

The inventory is written to `results/dataset_inventory.csv` and records the available filename, class, split assignment and video metadata/readability. Review these outputs before extraction, especially if a different HMDB51 archive or directory layout is used.

## 3. Environment

The project was developed on Windows with **Python 3.11.9**, **PyTorch 2.7.1+cu126**, **torchvision 0.22.1+cu126**, **Transformers 5.22.1**, and an NVIDIA GTX 1650 GPU (4 GB VRAM). Most classifier experiments operate on cached embeddings; video feature extraction is the GPU-intensive step.

Create and activate a Python 3.11 environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install a matching PyTorch/torchvision build using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) for your CPU/CUDA platform, then install the other dependencies:

```powershell
python -m pip install -r requirements.txt
python -m pip check
```

**Note:** `torch` and `torchvision` are documented separately in `requirements.txt` because the appropriate wheels depend on the host hardware and CUDA runtime. Model weights for DINOv2 and CLIP are downloaded by Transformers/Hugging Face on first use, so initial extraction/evaluation requires network access and sufficient disk space.

If PowerShell blocks venv activation, apply a temporary execution-policy change **only for the current shell** if permitted by your machine's security policy:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Implemented pipeline

1. **Inventory and validate** original video clips and official split assignments.
2. **Decode sequentially with OpenCV** to count decodable frames; sample **eight uniformly spaced** RGB observations per video.
3. **Extract frozen DINOv2-small features:** `facebook/dinov2-small`, final hidden-state CLS embedding, **384 features per frame**.
4. **Cache features:** one NPZ per clip for resumability, plus a consolidated float32 array with shape **`(600, 8, 384)`**.
5. **Fit supervised classifiers** using the official train partition (stratified train/validation separation) and evaluate on the held-out official test partition.
6. **Run controlled tests:** order reversal/shuffling, progressive observations, feature ablation, image corruption, central occlusion, confusion analysis and model-error comparison.
7. **Evaluate CLIP separately** as a zero-shot semantic baseline based on averaged frame-to-text similarity—not a temporal video-language model.

Model representations:

| Model | Input to classifier | What it tests |
|---|---|---|
| Single-frame MLP | Middle sampled frame, index 4; 384D | Static appearance baseline |
| Mean-pooling MLP | Average of eight frame embeddings; 384D | Additional visual evidence without order |
| Temporal GRU | Ordered `8 × 384` features; one GRU layer, 128 hidden units | Learned sequential processing |
| CLIP zero-shot | Mean frame-text similarity over eight frames | Semantic matching without task-specific training |

Feature scaling is fitted from training data only, using model-appropriate representations; it is not estimated from validation or test observations. The CLIP baseline uses its own image and text encoders and does not reuse DINOv2 embeddings.

For architecture diagrams, rationale and a **proposed (not implemented) larger multimodal perception system**, see [`docs/architecture.md`](docs/architecture.md).

## 5. Reproduce the workflow

**Run all commands from the project root.** Reproduction in order requires original dataset videos, split files, model downloads, and sufficient compute/storage.

### Step A — Generate and validate frame embeddings

```powershell
python -m src.embeddings.extract_embeddings
python scripts/validate_embeddings.py
```

Expected main artifact:

```text
data/processed/dinov2_hmdb51_split1.npz
```

The extraction code also writes resumable individual NPZ checkpoints and an extraction-failure log under `data/processed/`. The validation script checks shape, label and split counts, dtype, finiteness and other basic integrity constraints.

### Step B — Train the supervised baselines

```powershell
python -m src.models.single_frame_baseline
python -m src.models.mean_pool_baseline
python -m src.models.gru_temporal
```

These commands can overwrite existing checkpoints and metrics. The primary output files are:

```text
results/single_frame_metrics.json
results/mean_pool_metrics.json
results/gru_temporal_metrics.json
results/single_frame_baseline.pth
results/mean_pool_baseline.pth
results/gru_temporal.pth
```

### Step C — Run the temporal and robustness experiments

```powershell
python -m src.evaluation.temporal_order_test
python -m src.evaluation.progressive_observations
python -m src.evaluation.frame_ablation_test
python -m src.evaluation.visual_corruption_test
python -m src.evaluation.image_occlusion_test
```

Some controlled experiments, especially those that alter original images and re-extract features, require the raw videos and access to the pretrained DINOv2 model—not just the cached NPZ embeddings. Progressive-observation experiments train additional configurations and can regenerate checkpoints.

**Two progressive protocols are reported separately:** (A) first K chronological frames; and (B) coverage-controlled frame indices (`[4]`, `[0,7]`, `[0,2,5,7]`, all eight). They train separate GRU configurations, not a single online model updated frame by frame.

### Step D — Run CLIP zero-shot evaluation

```powershell
python -m src.vlm.clip_zero_shot
```

The experiment uses `openai/clip-vit-base-patch32` and six natural-language class prompts. It aggregates similarities from eight sampled frames. This is **zero-shot label matching**, not sequence description or temporal-change reasoning.

### Step E — Analyze and summarize

```powershell
python scripts/analyze_class_confusions.py
python scripts/compare_model_errors.py
python scripts/build_final_results_summary.py
```

Additional inspection and visualization utilities are in `scripts/` and `src/embeddings/`. Output directories include `results/class_confusion_analysis/`, `results/model_error_comparison/`, `results/temporal_analysis/`, `results/plots/` and the experiment-specific folders. Review produced figures and temporal frame strips as part of the assessment evidence.

## 6. Verified test results

All four approaches are reported on the same **180-video official split 1 test set**.

| Method | Test accuracy | Macro-F1 |
|---|---:|---:|
| Single-frame MLP | 69.44% | 69.14% |
| **Mean-pooling MLP** | **72.78%** | **72.64%** |
| Temporal GRU | 71.67% | 71.08% |
| CLIP zero-shot (8-frame score averaging) | 71.67% | 70.99% |

Mean pooling improves accuracy by **3.33 percentage points** over the single-frame MLP. The main GRU trails mean pooling by **1.11 percentage points**. The classifier architectures/training configurations are not perfectly matched; these are empirical comparisons, not a causal proof that order never matters.

### Evidence about chronological order

| Test on the trained GRU | Accuracy | Macro-F1 |
|---|---:|---:|
| Correct chronological order | 71.67% | 71.08% |
| Reversed order | 70.56% | 70.17% |
| Mean over 10 random frame-order shuffles | 71.78% | 71.27% |

Shuffled-order standard deviation: **1.52 pp accuracy / 1.47 pp macro-F1**, reflecting different permutations of the *same trained model*, **not** independent training runs. Correct ordering did not consistently outperform shuffled inputs in this experiment.

### Other verified findings

- **First-K GRUs (separately trained):** accuracy rises from **64.44%** with one initial observation to **72.78%** with all eight.
- **Coverage-controlled GRUs (separately trained):** one frame **73.89%**, two **70.56%**, four **73.89%**, eight **72.22%**. Performance is non-monotonic; the one-frame GRU cannot be directly compared with the one-frame MLP as a matched control.
- **Single-frame embedding ablation:** removing standardized frames 5 or 6 individually *increased* the main GRU's test accuracy to **73.89%**; this does not prove those frames are inherently useless.
- **Image-level corruptions:** Gaussian blur **69.44%**, Gaussian noise **67.78%**, half brightness **74.44%**, versus **71.67%** unmodified. Brightness improvement is specific to the tested condition and should not be generalized.
- **Central occlusion:** a black rectangle covering **25% of image width and 25% of height**, on every sampled frame, reduces accuracy to **68.33%**.
- **Shared errors:** among 180 videos, all three supervised classifiers are correct on the same **112** clips and wrong on the same **36**; the remaining **32** show mixed outcomes. The GRU repairs **7** errors made by mean pooling but introduces **9** errors on mean-pool successes.
- **Common confusions:** `clap → wave` and `drink → eat`. The GRU gets one more `drink` and one more `eat` example correct than mean pooling but performs worse on some other classes.

The full numerical results and their limitations are discussed in the [research report](docs/research_report.md). The measured gains support the usefulness of **multiple observations**; evidence of specifically **chronological temporal reasoning** remains limited.

## 7. Repository layout and artifact policy

```text
temporal-ai/
  README.md
  requirements.txt
  .gitignore
  docs/
    architecture.md
    research_report.md
    research_report.pdf
  src/
    embeddings/
      extract_embeddings.py
      analyze_temporal_embeddings.py
      inspect_temporal_results.py
    models/
      single_frame_baseline.py
      mean_pool_baseline.py
      gru_temporal.py
      checkpoint_utils.py
    evaluation/
      common.py
      temporal_order_test.py
      progressive_observations.py
      frame_ablation_test.py
      visual_corruption_test.py
      image_occlusion_test.py
    vlm/
      clip_zero_shot.py
  scripts/
    check_dataset_integrity.py
    inspect_video_dataset.py
    validate_embeddings.py
    analyze_class_confusions.py
    compare_model_errors.py
    build_final_results_summary.py
    visualize_sequences.py
    ...
  results/
    dataset_inventory.csv
    *_metrics.json
    class_confusion_analysis/
    model_error_comparison/
    clip_zero_shot/
    frame_ablation/
    visual_corruption/
    image_occlusion/
    temporal_analysis/
    plots/
    ...
  data/                    # local-only; ignored by Git
    raw/
    splits/
    processed/
```

The default `.gitignore` excludes `.venv/`, `data/`, and `*.pth` / `*.pt`. **As a result, model checkpoints are not automatically included in a Git submission.** The provided run commands allow models to be retrained once the dataset and dependencies are available. Small metrics, plots, failure-case tables and report files should remain available under `results/` in the final submission; confirm that they are actually tracked or included in the final archive.

Raw video archives, extracted videos, cached frame embeddings, per-video embedding checkpoints and the local virtual environment should not be included in a normal submission ZIP. Preserve all locally for potential reproduction and interview demonstration.

## 8. Reproducibility limits and future work

- The study covers **six of HMDB51's 51 classes**, **600** clips assigned to split 1 train/test, and **one official test split**.
- All representations use **eight uniformly sampled frames**; brief or direction-sensitive motions may be missed.
- Only one principal recurrent architecture was evaluated; model capacity and training procedures are not all matched.
- The single-frame model's validation split is reproducible, but its weight initialization and training stochasticity were **not fully seeded**. Do not assume exact metric replication on every fresh training run.
- Order-shuffle statistics come from **ten permutations**, not repeated independent model fits; the study has no training-run confidence intervals.
- Corruption severity, occlusion placement, and temporal perturbations are limited.
- CLIP is a **framewise zero-shot** baseline. The implemented study does **not** provide a generative VLM that narrates sequence changes.

Further work would focus on motion-aware frame selection, diagnostic motion-direction action pairs, matched temporal/non-temporal baselines, repeated seeds and paired uncertainty estimates, pose/flow features, and a video-language or multimodal semantic layer. The latter is an **architecture proposal**, not a claim of completed implementation.

## 9. Conclusion

This prototype shows that access to multiple visual observations improves action classification relative to one middle-frame observation. However, the tested GRU does not outperform mean pooling, and its accuracy is not consistently sensitive to preserving chronological order. The central takeaway is therefore a **distinction between multi-frame evidence and temporal reasoning**, rather than a claim that recurrent temporal modeling is universally ineffective.
