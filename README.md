# Explainability Examples

Hands-on demonstrations of techniques for explaining, inspecting, and attacking machine-learning models, from classic tabular feature attribution through image saliency, graph explanations, and LLM interpretability. Each example is self-contained and small enough to run and read in one sitting.

The examples answer three questions a production team gets asked about a model:

1. **Why did it make this prediction?** — feature attribution (SHAP, LIME, Grad-CAM, GNNExplainer)
2. **What has it learned inside?** — internals (attention, probing classifiers, sparse autoencoders, mechanistic interpretability)
3. **How do we see and document its behaviour?** — tooling and governance (TensorBoard, LLM observability, model cards), plus how fragile it is (adversarial examples)

---

## Contents

| File | Technique | Model / data | Framework | Runtime (CPU) |
|---|---|---|---|---|
| [`1-ExplainabilitySHAPhousing.ipynb`](1-ExplainabilitySHAPhousing.ipynb) | SHAP — global and local feature attribution | Random forest on California Housing | scikit-learn, shap | ~1 min (500-row sample) |
| [`2-ExplainabilityLimeCat.ipynb`](2-ExplainabilityLimeCat.ipynb) | LIME for images (superpixels) | MobileNetV2 (ImageNet) on `images/cat.jpg` | TensorFlow / Keras, lime | ~1 min |
| [`grad-cam.py`](grad-cam.py) | Grad-CAM saliency heatmap | ResNet-50 (ImageNet) on `images/cat.jpg` | PyTorch, torchvision | < 1 min |
| [`GNNExplainer.py`](GNNExplainer.py) | GNNExplainer for link prediction | 2-layer GCN on CiteSeer | PyTorch Geometric | 1–3 min |
| [`AdversarialMachineLearning.ipynb`](AdversarialMachineLearning.ipynb) | Adversarial examples (FGSM) | Dense network on MNIST | TensorFlow / Keras | ~1 min |
| [`MechInterp.py`](MechInterp.py) | Mechanistic interpretability with a pretrained sparse autoencoder | GPT-2 small + `gpt2-small-res-jb` SAE | TransformerLens, SAE Lens | 1–2 min (download on first run) |
| [`LMMExplain3.py`](LMMExplain3.py) | Nine LLM explainability approaches in one menu-driven script | DistilBERT, BERT, Claude API | transformers, shap, lime, anthropic | per example, seconds to minutes |
| [`tensorflow2.py`](tensorflow2.py) | TensorBoard basics: loss, predictions, model graph | Small MLP learning `sin(x)` | PyTorch + TensorBoard | seconds |
| [`tensorflowExpanded.py`](tensorflowExpanded.py) | TensorBoard showcase: multi-line scalars, images, histograms, figures, text | Synthetic data | PyTorch + TensorBoard | seconds |
| [`tensorboard1.py`](tensorboard1.py) | Knowledge-graph embeddings in the TensorBoard projector | TransE on Hetionet (`data/`) | AmpliGraph 2, TensorFlow | several minutes |

Supporting files:

| Path | What it is |
|---|---|
| `images/cat.jpg` | Input image for the LIME and Grad-CAM examples |
| `data/hetionet-v1.0-edges.sif.gz` | Hetionet biomedical knowledge graph edges (≈12 MB) for `tensorboard1.py` |
| `explain.png` | Saved output of `GNNExplainer.py` |
| `heatmap.jpg` | Saved output of `grad-cam.py` |
| `mech.png` | Saved output of `MechInterp.py` |
| `runs/` | Saved TensorBoard logs from `tensorflow2.py` and `tensorflowExpanded.py` — viewable without re-running |
| `instructor/INSTRUCTOR_GUIDE.md` | Teaching guide: preparation, session plans, expected outputs, exercises, and review questions |

---

## Setup

Python 3.10–3.12 is recommended. The examples span TensorFlow, PyTorch, and several interpretability libraries whose pins do not always agree, so **use separate virtual environments** rather than one environment for everything.

```bash
git clone git@github.com:AI-Demonstrations/explainability.git
cd explainability
```

### Environment A — tabular, image, adversarial (notebooks)

```bash
python -m venv .venv-tf && source .venv-tf/bin/activate
pip install jupyter ipywidgets pandas numpy matplotlib scikit-learn scipy shap \
            tensorflow lime scikit-image opencv-python
```

Covers `1-ExplainabilitySHAPhousing.ipynb`, `2-ExplainabilityLimeCat.ipynb`, and `AdversarialMachineLearning.ipynb`. Installing `ipywidgets` removes the `IProgress not found` warning that `shap` prints.

### Environment B — PyTorch (Grad-CAM, GNN, TensorBoard, LLM)

```bash
python -m venv .venv-torch && source .venv-torch/bin/activate
pip install torch torchvision tensorboard matplotlib numpy pillow scikit-learn
pip install torch_geometric                          # GNNExplainer.py
pip install transformers shap lime sentence-transformers \
            huggingface_hub anthropic python-dotenv  # LMMExplain3.py
pip install "transformer_lens>=4" "sae_lens>=6"      # MechInterp.py
```

`MechInterp.py` targets TransformerLens 4.x (`TransformerBridge`) and SAE Lens 6.x. Older releases (TransformerLens 3, SAE Lens 5) use a different API and will not run it. These libraries pin specific `torch`/`transformers` versions; if they conflict with the rest, give `MechInterp.py` its own environment.

### Environment C — knowledge-graph embeddings (optional)

`tensorboard1.py` needs **AmpliGraph 2**, which supports only a narrow range of TensorFlow versions. This combination is known to work:

```bash
python3.10 -m venv .venv-kg && source .venv-kg/bin/activate
pip install ampligraph==2.1.0 "tensorflow==2.10.*" "setuptools<81" pandas
```

AmpliGraph still imports `pkg_resources`, which setuptools 81 removed; hence the `setuptools<81` pin. See the [AmpliGraph documentation](https://docs.ampligraph.org/) for other version combinations.

### API key (LLM examples 3, 4, 8 only)

`LMMExplain3.py` reads the Anthropic key from `ANTHROPIC_API_KEY`, loading a `.env` file in the working directory if one exists. The older name `CLAUDKEY` is still accepted. If neither is set, it falls back to the SDK's own credential lookup, such as an `ant auth login` profile.

```bash
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env
```

`.env` is listed in `.gitignore`; never commit it.

### GPU

Everything runs on CPU. A GPU speeds up `MechInterp.py`, `GNNExplainer.py`, and the transformer examples, and they pick it up automatically (`cuda`, or `mps` on Apple silicon for the GNN example). The CUDA/cuDNN warnings TensorFlow prints at start-up are harmless.

Current PyTorch wheels no longer include kernels for older NVIDIA GPUs (Pascal, e.g. GTX 10-series, and earlier). On those cards `torch.cuda.is_available()` still returns `True`, but the scripts fail with `CUDA error: no kernel image is available for execution on the device`. Either hide the GPU so they run on CPU (`CUDA_VISIBLE_DEVICES= python GNNExplainer.py`), or install a PyTorch build that still supports your card (check `torch.cuda.get_arch_list()`). For Pascal cards, the CUDA 12.6 build works with every library above:

```bash
pip install "torch==2.14.1+cu126" "torchvision==0.29.1+cu126" --index-url https://download.pytorch.org/whl/cu126
```

---

## The examples

### 1. SHAP on a tabular regression — `1-ExplainabilitySHAPhousing.ipynb`

Trains a `RandomForestRegressor` on the California Housing dataset (20,640 block groups, 8 features, target = median house value), then compares two ways of ranking features:

- **Built-in random-forest importance** (impurity-based)
- **SHAP values** from `shap.Explainer`, which attribute each prediction to each feature using Shapley values from cooperative game theory

It then shows the main SHAP views:

| View | Scope | Question it answers |
|---|---|---|
| `shap.plots.bar(shap_values)` | Global | Which features matter most on average? |
| `shap.plots.beeswarm` / `summary_plot` | Global | How does a feature's *value* push predictions up or down? |
| `summary_plot(plot_type='violin')` | Global | Distribution of each feature's impact |
| `shap.plots.bar(shap_values[0])` | Local | Which features drove *this one* prediction? |
| `shap.plots.waterfall` | Local | Step from the base value to this prediction |
| `shap.plots.force` | Local | The same, as a single push/pull bar (static image; the interactive version is in a comment) |

**What to look for:** both methods rank `MedInc` (median income) first, but they disagree after that. The forest's impurity importance puts `AveOccup` second, while SHAP ranks `Latitude` and `Longitude` well above it. Impurity importance is known to be biased, and SHAP measures the effect on predictions directly. This disagreement is the main teaching point of the notebook.

Expected results (the split and forest use `random_state = 42`, so they repeat): R² ≈ 0.80, MAE ≈ 0.33. The SHAP importances are also printed as shares that add up to 1, like the forest's own importances, so the two rankings can be compared directly.

> **Runtime:** the notebook explains a 500-row sample of the test set (`N_EXPLAIN = 500`), which takes about a minute (the forest uses every CPU core via `n_jobs = -1`). Set `N_EXPLAIN = None` to explain all ~4,100 rows; that takes about 8 times as long. For tree models, `shap.TreeExplainer(model)` is exact and takes seconds.

### 2. LIME for images — `2-ExplainabilityLimeCat.ipynb`

Classifies `images/cat.jpg` with an ImageNet-pretrained MobileNetV2, then uses `lime_image.LimeImageExplainer` to find which **superpixels** (image regions) support the top prediction. LIME switches regions on and off, watches how the prediction changes, and fits a simple linear model to those perturbations.

**What to look for:** the top-3 predictions are `lynx` (29%), `plastic_bag` (15%), `tiger_cat` (9%). The model is not confident, and the LIME mask shows whether it is looking at the cat or the background. That makes it a good example of an explanation exposing a weak model.

Notes:
- `num_samples=100` keeps it fast; real use needs 1,000+ for stable masks.
- A TensorFlow Grad-CAM implementation is included in the notebook as a commented-out block. The PyTorch version in `grad-cam.py` is the one that runs.

### 3. Grad-CAM — `grad-cam.py`

Grad-CAM weights the last convolutional layer's feature maps by the gradient of the target class, giving a coarse heatmap of *where* the CNN looked. This version uses ResNet-50 and `layer4[-1]` as the target layer, with forward/backward hooks.

```bash
python grad-cam.py
```

It reads `images/cat.jpg`, prints the model's top 5 predictions, and writes `heatmap.jpg` next to the script, so it runs from any directory. The figure shows the original image beside a Grad-CAM heatmap for each of the top two predictions, with a colour bar. The colours run from blue (least support for that class) to red (most). Each heatmap is scaled to 0–1 on its own, so the colours are relative within one heatmap, not probabilities.

**What to look for:** ResNet-50's top guess is `plastic bag` (9.9%), just ahead of `tabby` (8.8%). The "plastic bag" heatmap lights up the white petals below the cat, while the "tabby" heatmap lights up the cat's face. An explanation answers "why *this* class?", so always check which class it explains. To explain a different class, change `target_classes` in the script, e.g. `[categories.index("tabby")]`.

Compare `heatmap.jpg` with the LIME mask. The two methods often highlight different regions for the same image.

### 4. GNNExplainer — `GNNExplainer.py`

Trains a two-layer GCN for **link prediction** on the CiteSeer citation graph (validation ROC-AUC printed every 20 epochs), then runs `GNNExplainer` to find which edges and node features matter most for one predicted link. It writes the explanation subgraph to `explain.png`.

```bash
python GNNExplainer.py
```

CiteSeer downloads automatically to `data/Planetoid/` inside the repo on the first run (ignored by git).

### 5. Adversarial examples (FGSM) — `AdversarialMachineLearning.ipynb`

Trains a small dense network on MNIST (~97% test accuracy), then applies the **Fast Gradient Sign Method**, which nudges every pixel by ε = 0.1 in the direction that increases the loss.

**What to look for:** the perturbed "7" is classified as a different digit (which one varies between training runs, often "2" or "3"), and accuracy on the full adversarial test set drops from **97.4% to about 3%** even though the images still look the same to a person. This is why explainability and robustness are taught together: a model can be accurate and still rely on features no human would use.

### 6. Mechanistic interpretability — `MechInterp.py`

Loads GPT-2 small through TransformerLens, captures the residual stream at `blocks.8.hook_resid_pre` with a hook, and encodes it with a pretrained **sparse autoencoder** (`gpt2-small-res-jb` from SAE Lens). It prints the top 20 most active SAE features for the input sentence, plots which features fire at each token position, and reports overall sparsity.

```bash
python MechInterp.py
```

Look up any printed feature index on [Neuronpedia](https://www.neuronpedia.org/gpt2-small/res-jb) to see what concept it represents. The sparsity plot is saved to `mech.png`.

### 7. LLM explainability tour — `LMMExplain3.py`

One script with nine short, independent examples, each with a docstring explaining what it shows and its limitations:

| # | Example | What it shows | Needs API key |
|---|---|---|---|
| 1 | SHAP | Token-level attribution for a DistilBERT sentiment classifier | |
| 2 | LIME | Word-masking explanation for the same classifier | |
| 3 | Chain-of-thought | Visible step-by-step reasoning from Claude, and why it may not reflect internal computation | ✔ |
| 4 | RAG transparency | Retrieval with `all-MiniLM-L6-v2`, then a Claude answer citing its sources | ✔ |
| 5 | Attention | BERT attention weights per layer and head (bertviz hook included, commented out) | |
| 6 | Probing classifiers | Logistic-regression probes on each BERT layer's hidden states | |
| 7 | Sparse autoencoder | A small SAE trained from scratch, showing the mechanism `MechInterp.py` uses with a pretrained SAE | |
| 8 | LLM observability | Tracing latency, tokens, and simple hallucination-risk signals around Claude calls | ✔ |
| 9 | Model card | Writing and parsing a Hugging Face model card for governance | |

```bash
python LMMExplain3.py          # list the examples
python LMMExplain3.py 6        # run one
python LMMExplain3.py all      # run all nine (3, 4, 8 need an API key)
```

Examples 3, 4 and 8 call Claude through one helper, `_ask_claude()`, using the model named in `CLAUDE_MODEL` at the top of the file (`claude-opus-5-5`). To change models, edit that one line. The helper:

- runs at **low effort** to keep the demos fast and cheap;
- returns only the answer text. Current Claude models think before answering, so a response can contain thinking blocks as well as text;
- enables the API's **server-side fallback**: if a safety classifier declines a prompt, the request is retried on another model within the same call.

On example 3: because current models also reason internally, the "direct" answer is not reasoning-free. What chain-of-thought adds is reasoning that is *visible* in the answer, and that visible part is what a human can inspect and audit.

### 8. TensorBoard — `tensorflow2.py`, `tensorflowExpanded.py`, `tensorboard1.py`

Training visibility is the first explainability tool most teams use. Despite the file names, the first two scripts use **PyTorch's** `SummaryWriter`.

```bash
python tensorflow2.py          # writes runs/neural_network_sine
python tensorflowExpanded.py   # writes runs/tensorboard_showcase
tensorboard --logdir runs      # open http://localhost:6006
```

- `tensorflow2.py` logs training loss, the model's prediction of `sin(π/2)` converging to 1.0, and the model graph.
- `tensorflowExpanded.py` shows each TensorBoard tab: overlaid train/validation curves, an image grid, weight histograms over time, a Matplotlib figure, and Markdown text.
- `tensorboard1.py` trains TransE embeddings on Hetionet and exports them for the **Projector** tab, so you can explore which entities cluster together:

  ```bash
  python tensorboard1.py
  tensorboard --logdir ./embeddings_vis --port 8891
  # open http://127.0.0.1:8891/#projector
  ```

  It reads `data/hetionet-v1.0-edges.sif.gz` relative to the script and writes the projector files to `./embeddings_vis`.

The saved logs in `runs/` let you open TensorBoard immediately without running anything.

---

## Suggested order

1. **SHAP** (sampled) → **LIME** → **Grad-CAM**: the same question, "why this prediction?", answered three ways, on tabular data and then images.
2. **Adversarial examples**: why a plausible explanation doesn't guarantee a robust model.
3. **GNNExplainer**: attribution on graph-structured data.
4. **LMMExplain3.py 1, 2, 5, 6** → **MechInterp.py**: from treating the LLM as a black box to looking inside it.
5. **LMMExplain3.py 3, 4, 8, 9** and **TensorBoard**: the techniques a production system actually runs and documents.

## Choosing a technique

| You need to… | Use | Caveat |
|---|---|---|
| Explain a tabular model, globally and per prediction | SHAP (`TreeExplainer` for trees) | Exact/kernel explainers are slow; correlated features share credit |
| Explain any black-box model on one input | LIME | Unstable between runs; depends on how inputs are perturbed |
| See where a CNN looked | Grad-CAM | Coarse resolution; needs a convolutional layer |
| Explain a GNN prediction | GNNExplainer | Explanation is itself an optimisation and can vary between runs |
| Understand what a transformer represents | Probing, SAEs, TransformerLens | Research-grade; findings are about representations, not decisions |
| Make an LLM's answer auditable | RAG with citations, observability traces | Citations show sources, not reasoning |
| Show reasoning to a user | Chain-of-thought | Reasoning shown may not be faithful to the computation |
| Document a model for governance | Model card | Only as good as the evaluation behind it |

---

## Notes

- Saved outputs in the notebooks come from earlier runs and may differ slightly from a fresh run (random train/test split, sampled SHAP rows). In particular, the SHAP notebook's saved plots were produced from the full test set; re-run it to get plots for the 500-row sample.
- `AdversarialMachineLearning.ipynb` was previously named `AdversarialMachineLearming.ipynb`. The older duplicate notebooks (`ExplainabilitySHAPhousing.ipynb`, `ExplainabilityLimeCat.ipynb`, `cat1.ipynb`) have been removed; they remain in git history.

## References

- Lundberg & Lee, *A Unified Approach to Interpreting Model Predictions* (SHAP), NeurIPS 2017
- Ribeiro, Singh & Guestrin, *"Why Should I Trust You?": Explaining the Predictions of Any Classifier* (LIME), KDD 2016
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, ICCV 2017
- Ying et al., *GNNExplainer: Generating Explanations for Graph Neural Networks*, NeurIPS 2019
- Goodfellow, Shlens & Szegedy, *Explaining and Harnessing Adversarial Examples* (FGSM), ICLR 2015
- Bricken et al., *Towards Monosemanticity: Decomposing Language Models With Dictionary Learning*, Anthropic, 2023
- Mitchell et al., *Model Cards for Model Reporting*, FAT* 2019
- Himmelstein et al., *Systematic integration of biomedical knowledge prioritizes drugs for repurposing* (Hetionet), eLife 2017
