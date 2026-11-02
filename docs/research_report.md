Temporal & Multimodal Visual Intelligence
Technical research report — six-class HMDB51 action recognition. Experimental results below were checked against the project's saved summary metrics, perturbation results, confusion matrices, and model-error comparisons. The accompanying PDF is the condensed six-page submission edition.

1. Research question and central finding
Can a model use information from a sequence of visual observations that is not available in one frame? I compare one-frame recognition, unordered eight-frame aggregation, and ordered recurrent modeling using a common frozen DINOv2 representation. An independent CLIP image-text baseline probes semantic zero-shot labeling.

Principal finding: Eight-frame mean pooling achieved 72.78% accuracy / 72.64% macro-F1, versus 69.44% / 69.14% for the single-frame MLP. The temporal GRU achieved 71.67% / 71.08% and did not beat unordered pooling. Across ten random order perturbations, the GRU averaged 71.78% accuracy. Thus multiple visual observations help in this setting, but a benefit from chronological relationships has not been established.

2. Dataset, frame extraction, and embeddings
I selected six categories from HMDB51: brush_hair, clap, drink, eat, pour, and wave. From the selected subset, 600 videos participate in official split 1: 420 training and 180 test clips, evenly divided at 70 train / 30 test per class. I stratified the official training set into 336 training / 84 validation clips for classifier selection; the official test set remained separate.

I decoded each video sequentially using OpenCV. After counting decodable frames, I generated eight approximately uniform temporal indices, decoded the video again and converted sampled BGR frames to RGB. I then applied the pretrained image processor for frozen facebook/dinov2-small, saving the final hidden state's CLS token as a 384-dimensional vector per frame. Extraction used inference mode and four-frame batches. Per-video NPZ checkpoint files were consolidated into a (600, 8, 384) float32 embedding dataset with labels, filenames, official splits, and encoder metadata. Each classifier used its own scaler fitted on its training representation only.

The resulting ordered trajectory can be written as E = [e₁, …, e₈] ∈ ℝ⁸ˣ³⁸⁴. This makes adjacent-frame cosine similarity, between-video similarity, and temporal variation possible to inspect. The implementation contains temporal embedding-analysis utilities; I do not ascribe unverified numeric similarity measurements to them here.

3. Models and system architecture
The implemented pipeline and proposed multimodal extension are diagrammed in [`temporal_ai_architecture.md`](temporal_ai_architecture.md). All supervised classifiers share the same frozen DINOv2 frame embeddings but differ in how they aggregate evidence:
- Single-frame MLP: one 384D middle-frame embedding (frame index 4).
- Mean-pooling MLP: arithmetic mean of all eight 384D embeddings; order-invariant.
- Temporal GRU: ordered sequence of eight embeddings, single recurrent layer with hidden dimension 128, then classification head.
- Zero-shot CLIP: openai/clip-vit-base-patch32, six textual action prompts, frame-wise image-text similarity averaged across eight frames. No supervised task-specific fitting and no explicit temporal language processing.
I evaluate all methods on the same official 180-video test partition using accuracy, macro-F1, and class-wise confusion. Shared representation reduces a major confound, but classifier architectures and training regimes are not perfectly parameter matched.

4. Main results and errors
Model	Evidence used	Accuracy	Macro-F1
Single-frame MLP	Middle frame	69.44%	69.14%
Mean-pooling MLP	Unordered mean of 8 frames	72.78%	72.64%
Temporal GRU	Ordered 8-frame sequence	71.67%	71.08%
CLIP zero-shot	Mean of frame-text scores	71.67%	70.99%


Mean pooling improves accuracy by 3.33 percentage points (macro-F1 +3.50 pp) versus one frame; the GRU is 1.11 pp less accurate (macro-F1 −1.56 pp) than mean pooling. CLIP matches GRU accuracy but provides a semantic, order-agnostic comparison rather than a video-language explanation of change.
Correctly identified test clips per class (out of 30)
True class	Single frame	Mean pool	GRU
brush_hair	25	27	25
clap	12	13	12
drink	18	20	21
eat	21	22	23
pour	27	27	26
wave	22	22	22
Total / 180	125	131	129


The GRU gains one correct clip each for drink and eat relative to mean pooling, but loses on brush_hair, clap, and pour. Major recurring GRU confusions include clap → wave (11) and drink → eat (8), consistent with overlapping hand motion or hand–object appearance. All three supervised models correctly classify 112 of the same 180 videos and misclassify 36 of the same videos; the remaining 32 have mixed predictions. Against mean pooling, the GRU fixes 7 errors but introduces 9 errors.

5. Controlled temporal evaluations
5.1 Order sensitivity — same trained GRU
Input order	Accuracy	Macro-F1
Original	71.67%	71.08%
Reversed	70.56%	70.17%
Mean across ten random shuffles	71.78%	71.27%
Standard deviation of shuffles	1.52 pp	1.47 pp


Reversing order decreases GRU accuracy by 1.11 pp, while the ten shuffled sequences range from 70.00% to 75.00% accuracy and average slightly above the original order. The changes are modest and inconsistent; the shuffle standard deviation measures variability across input permutations not variability across training runs. These results do not provide strong evidence that correct chronological order is essential to the classifier.
5.2 Progressive observations — two distinct protocols
Protocol A: first K chronological frames. Separately trained GRUs use prefixes of the original eight-frame observation sequence.
K	1	2	4	8
Accuracy	64.44%	70.00%	70.56%	72.78%
Macro-F1	64.04%	69.72%	70.05%	72.27%


Protocol B: coverage-controlled frames. Separately trained GRUs use indices distributed through the clip.
K	Sampled indices	Accuracy	Macro-F1
1	[4]	73.89%	73.64%
2	[0, 7]	70.56%	69.74%
4	[0, 2, 5, 7]	73.89%	72.98%
8	[0, 1, 2, 3, 4, 5, 6, 7]	72.22%	71.26%


The first-K study shows increased accuracy with longer prefixes, whereas the coverage-controlled study is non-monotonic. Both are separate training configurations, not progressive inference through a single model; progressive GRUs also have a somewhat different classification head from the principal GRU. The one-frame coverage-controlled GRU result must not be treated as a controlled comparison with the single-frame MLP.
5.3 Frame-level standardized embedding ablation
I replaced one standardized frame embedding at a time with a zero vector, representing its featurewise training mean after scaling. The original accuracy was 71.67%. Ablating frame 1 produced 71.11%; frames 5 and 6 each produced 73.89% (+2.22 pp); frame 7 produced 73.33% (+1.67 pp). Some frames may contribute redundant or less-useful information to the trained model, but this does not show that later observations are intrinsically useless or that removing multiple frames would have the same effect.

6. Robustness, embedding interpretation and CLIP
Test-time condition	Accuracy	Macro-F1	Change in accuracy
Unmodified	71.67%	71.08%	—
Gaussian blur, radius 2	69.44%	69.07%	−2.22 pp
Gaussian noise, σ=20 on 0–255 pixel values	67.78%	67.40%	−3.89 pp
Brightness × 0.5	74.44%	74.18%	+2.78 pp
Central black occlusion, 25% width × 25% height	68.33%	67.54%	−3.33 pp


All image-level interventions were applied before extracting DINOv2 embeddings and used the same trained GRU. Noise reduced accuracy the most among tested conditions; spatial masking and blur also harmed it. Unexpected improvement under reduced brightness may be an incidental, test-specific feature effect, not evidence that darker images universally help. The study tested one severity per corruption and one mask position and size.

I evaluated the CLIP zero-shot model on all 180 official test videos with zero recorded failures. Its 71.67% accuracy / 70.99% macro-F1 demonstrates useful semantic alignment between class phrases and visual content, without supervised task-specific training. However, because the implementation averages frame-level similarity scores, it cannot directly explain when new action information appears or describe transitions between observations. This motivates a video-language model or a temporal semantic layer in the proposed system architecture.

7. Direct answers to the assessment research questions
A. When did temporal information help? Additional observations helped in the mean-pool-versus-single-frame comparison (+3.33 pp). A separately trained first-K GRU also improved as the chronological prefix grew, while the GRU gained slight class-wise recognition of drink and eat. These findings concern multi-frame evidence and do not prove dependence on frame order.

B. When did it not help? The temporal GRU did not beat mean pooling (71.67% versus 72.78%); shuffled order performed similarly to original order; coverage-controlled GRUs did not improve monotonically with more frames. Confusion between clap and wave persisted.

C. How can we test whether a temporal model uses relationships rather than simply more visual information? Preserve frame content and embeddings while perturbing order; compare to order-invariant aggregation using matched data, model capacity, and training; evaluate motion-direction-dependent examples; repeat experiments across random seeds and analyze per-example changes. The current order-perturbation experiment gives limited evidence for strong chronological dependence.

D. What would convince me the temporal model is not useful for this task? Across repeated independent runs and temporally diagnostic held-out cases, an appropriately tuned, comparable sequential model would fail to beat pooling while remaining insensitive to systematic order changes. The present evidence trends toward that result but is not yet sufficiently replicated to establish it decisively.

E. What would provide stronger evidence? Increase sample and class coverage, use additional official splits, intentionally include time-direction-sensitive action pairs, compare uniform to motion-aware or denser frame sampling, test pose or optical-flow features, and report repeated-seed uncertainty and paired statistical comparisons. Inspect real frame strips and individual errors to determine when discriminative information first appears.

8. Limitations, priorities and conclusion
Limitations include the restricted six-class / 600-video subset and 180-video test set, eight uniform frames, a single main GRU architecture, one official test split, modest mismatches in training/model heads, limited corruption severity and occlusion placement, only ten frame permutations, and incomplete seeding of the single-frame MLP. With only 180 test videos, one prediction corresponds to approximately 0.56 percentage points, so small gaps should not be overstated.

If I had two additional days, I would prioritize: (1) diagnostic action pairs and motion-aware frame selection with matched baselines; (2) repeated seeds, confidence intervals and paired comparisons; (3) reusable configurations and tests, frame-strip/embedding plots and reproducible error-case documentation.

Conclusion. The experiment supports the value of multiple visual observations, but it does not demonstrate that the tested recurrent model extracts additional benefit from their chronological structure. That distinction is the principal research finding.