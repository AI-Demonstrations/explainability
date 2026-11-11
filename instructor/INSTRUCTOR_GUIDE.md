# Instructor's Guide — Explainability Examples

This guide is for whoever teaches from this repository. The top-level [README](../README.md) explains what each example is and how to install it. This guide covers how to teach them: what to prepare, what to say, what students should notice, what goes wrong in class, and how to check understanding.

All expected outputs and timings below come from a test run in November 2026 on CPU, with current library versions (PyTorch 2.15, TensorFlow 2.21, transformers 5.22, TransformerLens 4.3, SAE Lens 6.55). Numbers that depend on random seeds are marked "varies".

---

## 1. Course goals

By the end, students should be able to:

1. Explain the difference between **global** explanations (how the model behaves overall) and **local** ones (why it made one prediction).
2. Pick a suitable technique for a model type and a question (tabular, image, graph, text, or LLM).
3. Run SHAP, LIME, Grad-CAM and GNNExplainer, and read their output correctly.
4. Explain why two explanation methods can disagree, and why neither is "the truth".
5. Describe what looking inside a model means (attention, probing, sparse autoencoders) and what it can't tell you.
6. Show how a model can be accurate and still fragile (adversarial examples).
7. Name the explainability practices a production LLM system actually uses: citations, tracing, and model cards.

The theme to keep coming back to: **an explanation is itself a model of the model, and it can be wrong.** Every example has a limitation worth naming out loud.

---

## 2. Before class

### 2.1 Environments (do this days ahead, not on the morning)

The examples need three separate Python environments because their dependencies conflict. Follow README § Setup. Allow about 30 minutes and ~10 GB of disk for the downloads.

| Environment | Covers | Python |
|---|---|---|
| A: TensorFlow | Both notebooks numbered 1–2, adversarial notebook | 3.10–3.12 |
| B: PyTorch | Grad-CAM, GNN, TensorBoard scripts, `LMMExplain3.py`, `MechInterp.py` | 3.12 |
| C: AmpliGraph | `tensorboard1.py` only (optional) | **3.10** |

Known traps:
- `MechInterp.py` needs **TransformerLens 4+ and SAE Lens 6+**. Older versions fail on import.
- AmpliGraph needs `setuptools<81`, or it fails with `No module named 'pkg_resources'`.
- **Older NVIDIA GPUs** (GTX 10-series and earlier) are not supported by current PyTorch. The scripts detect the GPU and then crash with `no kernel image is available`. Either install the CUDA 12.6 PyTorch build given in README § GPU, which supports these cards, or run with `CUDA_VISIBLE_DEVICES=` to force CPU. Everything in this course runs fine on CPU.

### 2.2 Pre-download models and data

The first run of several examples downloads models or datasets. Do it once before class so a slow network doesn't stall the session:

```bash
# Environment B
python grad-cam.py              # ResNet-50 weights (~100 MB)
python GNNExplainer.py          # CiteSeer dataset
python LMMExplain3.py 1         # DistilBERT sentiment model
python LMMExplain3.py 4         # all-MiniLM-L6-v2 (needs API key)
python LMMExplain3.py 5         # bert-base-uncased
python MechInterp.py            # GPT-2 small + SAE weights (~1 GB)

# Environment A: run each notebook once (MobileNetV2, MNIST, California Housing)
```

### 2.3 API key (for LLM examples 3, 4 and 8)

`LMMExplain3.py` examples 3, 4 and 8 call Claude. Put the key in `.env` next to the script:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Common problems:

| Symptom | Cause |
|---|---|
| "No Anthropic API key found" | `.env` missing, in another directory, or `python-dotenv` not installed (the script now warns about this) |
| "starts with 'sk-proj'…" | An OpenAI key was pasted instead of an Anthropic one |
| `401 invalid x-api-key` | The key was revoked, disabled, or copied incompletely; create a new one in the Anthropic Console |
| Nothing loads, no error | The line is missing its `=` sign |

To check a key without spending tokens, use `client.models.list(limit=1)`.

**Never show `.env` on a projector.** If a key appears on screen or in a recording, revoke it straight after class. Cost for a full run of examples 3, 4 and 8 is a few cents.

### 2.4 Fallbacks if something breaks live

- The notebooks keep saved outputs from earlier runs, so you can teach from them without re-running.
- `heatmap.jpg`, `explain.png` and `mech.png` are saved outputs of Grad-CAM, GNNExplainer and MechInterp.
- `runs/` holds saved TensorBoard logs: `tensorboard --logdir runs` works without running anything.
- If the API key fails, show the expected outputs in § 4.7 of this guide.

---

## 3. Suggested session plans

### Option 1: one 3-hour session

| Time | Block | Examples |
|---|---|---|
| 0:00 | Framing: why explain models? Global vs local | (slides or whiteboard) |
| 0:15 | Tabular attribution | SHAP notebook |
| 0:45 | Image attribution, two methods | LIME notebook → `grad-cam.py` |
| 1:15 | Robustness | Adversarial notebook |
| 1:35 | *Break* | |
| 1:45 | Text models: attribution and internals | `LMMExplain3.py` 1, 2, 5, 6 |
| 2:15 | Mechanistic interpretability | `LMMExplain3.py` 7 → `MechInterp.py` |
| 2:35 | Production LLM practice | `LMMExplain3.py` 3, 4, 8, 9 |
| 2:55 | Wrap-up: choosing a technique | README § Choosing a technique |

Graphs (`GNNExplainer.py`) and TensorBoard are left out; set them as homework or a lab.

### Option 2: three 90-minute sessions

1. **Why this prediction?** SHAP → LIME → Grad-CAM → GNNExplainer.
2. **Can we trust it?** Adversarial examples, TensorBoard (training visibility), then LLM attribution (examples 1, 2).
3. **Inside the model, and in production.** Attention, probing, SAEs, `MechInterp.py`, then chain-of-thought, RAG, observability, model cards.

---

## 4. Teaching notes per example

Each section gives the key idea, what to run, what students should see, the point to make, and questions to ask.

### 4.1 SHAP on tabular data — `1-ExplainabilitySHAPhousing.ipynb`

**Key idea.** Shapley values from game theory split a prediction fairly among the input features. Feature values push the prediction up or down from a baseline (the average prediction).

**Runtime.** About 1 minute (500-row sample). Start it before you begin talking.

**Expected output** (fixed random seed, so it should match closely).
- R² ≈ 0.80, MAE ≈ 0.33.
- Random-forest impurity importance: MedInc 0.52, **AveOccup 0.14**, Latitude 0.09, Longitude 0.09, …
- SHAP importance (mean |SHAP|): MedInc 0.51, **Latitude 0.37, Longitude 0.29**, AveOccup 0.21, …
- The same SHAP numbers as shares of the total: MedInc 33%, Latitude 24%, Longitude 19%, AveOccup 13%. Both sets of shares add up to 1, so put them side by side: the forest gives income 52%, SHAP gives it 33%.

**The point to make.** Both methods put median income first, but then they disagree. Impurity importance is known to favour features with many distinct values. SHAP measures each feature's effect on the predictions directly. Location (latitude and longitude together) matters far more than the forest's own ranking suggests, which matches what anyone knows about California house prices.

**Walk through the plots in this order:** bar (global ranking) → beeswarm (direction: high income pushes price up) → waterfall for one house (local) → force plot (the same, as one push/pull bar).

**Questions to ask.**
- Why might the two rankings disagree? *(Impurity importance is biased toward high-cardinality features; SHAP looks at prediction effects.)*
- Latitude and longitude are strongly related to each other. What does that do to SHAP? *(Correlated features share credit, so each one alone can look less important than the pair is.)*
- The notebook uses `shap.Explainer(model.predict, …)`. Why is that slow, and what would be faster? *(It treats the model as a black box; `TreeExplainer` uses the tree structure and is exact and fast.)*

### 4.2 LIME for images — `2-ExplainabilityLimeCat.ipynb`

**Key idea.** Hide parts of the input (here, superpixels), see how the prediction changes, and fit a simple linear model to those results. The explanation is that simple model.

**Runtime.** About 10 seconds.

**Expected output.** Top-3: **lynx 29%, plastic_bag 15%, tiger_cat 9%**, then a mask over the image.

**The point to make.** The model isn't confident, and "plastic bag" is a strange second guess. The LIME mask shows whether the model looked at the cat or at the background. This is a good example of an explanation exposing a weak model, which is one of the most useful things explanations do in practice.

**Demonstration worth doing.** Re-run the LIME cell two or three times. With `num_samples=100` the mask changes between runs. Ask: *if the explanation changes each time, which one is right?* Then raise `num_samples` to 1000 and show it settling down.

**Questions to ask.**
- What does `hide_color=0` mean, and why might the choice matter? *(Hidden regions become black; the model's reaction to black patches is part of the explanation.)*
- LIME fits a linear model. When would a linear approximation be misleading? *(When features interact strongly, as pixels do.)*

### 4.3 Grad-CAM — `grad-cam.py`

**Key idea.** For a CNN, take the last convolutional layer's feature maps and weight each one by how much the target class's score depends on it (the gradient). The result is a coarse heatmap of where the network looked.

**Runtime.** A few seconds. Prints the top 5 predictions and writes `heatmap.jpg`: the photo, then one heatmap for each of the top two classes, with a colour bar.

**Expected output.** Top 5: **plastic bag 9.9%, tabby 8.8%**, daisy 7.3%, lynx 7.2%, Persian cat 5.9%. The "plastic bag" heatmap is hottest on the white petals below the cat; the "tabby" heatmap is hottest on the face.

**Reading the colours.** Blue → green → yellow → red means least → most support for that class. Each heatmap is rescaled to 0–1 on its own, so there is always a red area even when the evidence is weak, and blue means "no support", not "evidence against" (the ReLU drops negative evidence).

**The point to make first.** The model's top guess is wrong and barely ahead of "tabby". Each heatmap answers "why *this* class?", and the two point at different parts of the photo. Before reading any explanation, check which prediction it explains.

**The point to make next.** Put `heatmap.jpg` next to the LIME mask. They were produced by **different models** (ResNet-50 vs MobileNetV2) and **different methods**, and they often highlight different regions. Ask students which one to trust. The honest answer is that each is a partial view.

**Code walk-through (5 minutes, worth it).** Show the two hooks in `generate_grad_cam`: the forward hook saves activations, the backward hook saves gradients. This is the same hook mechanism `MechInterp.py` uses later, so it's a good bridge.

**Questions to ask.**
- Why is the heatmap so blocky? *(ResNet-50's last conv layer is 7×7, so the map is upscaled from 49 cells.)*
- Can Grad-CAM be used on a transformer or a random forest? *(Not directly; it needs convolutional feature maps.)*

### 4.4 Adversarial examples (FGSM) — `AdversarialMachineLearning.ipynb`

**Key idea.** Nudge every pixel slightly (ε = 0.1) in the direction that most increases the loss. The change is barely visible to a person but changes the model's answer.

**Runtime.** About 20 seconds.

**Expected output.** Normal test accuracy **≈ 97.5%**; adversarial accuracy **≈ 2–3%**. The perturbed "7" is classified as a different digit (which one varies between runs; a "2" or a "3" is common).

**The point to make.** This is why explainability and robustness belong together. A model can be highly accurate and still rely on patterns no human would use. A plausible-looking explanation doesn't guarantee a robust model.

**Questions to ask.**
- Why does taking the *sign* of the gradient work so well? *(It moves every pixel by the maximum allowed amount in the harmful direction.)*
- How could you defend against this? *(Adversarial training, input preprocessing, certified defences; and the attack and defence keep escalating.)*
- Would SHAP or LIME explanations of the adversarial image look strange? Try it as an exercise.

### 4.5 GNNExplainer — `GNNExplainer.py`

**Key idea.** For a graph neural network, learn a small mask over edges and node features that keeps the prediction the same. The edges that survive are the explanation.

**Runtime.** About 40 seconds on CPU. Writes `explain.png`.

**Expected output.** Validation ROC-AUC printed every 20 epochs, around **0.73**, then the explanation subgraph.

**Teaching moment.** Training loss drops to nearly zero while validation AUC stops improving. The model is **overfitting**. That's worth pointing out: GNNExplainer will still produce a confident-looking explanation of an overfitted model. Explanations describe what the model does, not whether it's any good.

**Questions to ask.**
- The explanation is the result of an optimisation. What does that imply if you run it twice? *(It can differ; check stability before trusting it.)*
- Why is explaining a graph prediction harder than explaining a tabular one? *(The input isn't a fixed set of features; the relevant structure is a neighbourhood.)*

### 4.6 LLM attribution and internals — `LMMExplain3.py` 1, 2, 5, 6, 7

Run each with `python LMMExplain3.py <n>`.

**1 — SHAP on text.** DistilBERT sentiment on "The movie was surprisingly good but a bit long." Expect **"but" to have the largest negative value (≈ −0.47)** and "long" to be negative. Point: the word "but" signals that what follows outweighs what came before. The model has learned that.

**2 — LIME on text.** "The film had great acting but a terrible plot." Expect **"terrible" strongly negative (≈ −0.59)**, "great" positive (≈ +0.25), and "but" negative again. Compare with example 1: two methods, same lesson about "but". With only 100 samples the numbers shift between runs.

**5 — Attention.** BERT on "The animal didn't cross the street because it was too tired." In the last layer, averaged across heads, **nearly every token attends most to the full stop**. Students often expect "it" to attend to "animal".

*The point to make:* averaged attention is often dominated by "sink" tokens like punctuation and `[SEP]`. Attention weights are not explanations of causal importance. Individual heads in middle layers are more interesting; uncomment the bertviz block in a notebook to explore them.

**6 — Probing classifiers.** Trains a logistic regression on each BERT layer to tell nouns from verbs. Expect **100% accuracy at every layer, including layer 0**.

*Use this as a lesson about probing, not a result.* With 10 examples and 768-dimensional features, a linear probe can separate almost anything, even raw word embeddings before any context. A probe result only means something with enough data and a control task (for example, random labels) to show the probe can't just memorise. Ask students how they would design a better experiment.

**7 — Sparse autoencoder from scratch.** A small SAE learns to represent simulated activations with a larger, sparse set of features. Expect reconstruction error to fall and **about 35% of features active** by the end. That's not very sparse; real SAEs aim for a few percent (compare `MechInterp.py`, which shows 99.8% zeros). Raising `sparsity_coeff` is a good exercise.

### 4.7 Mechanistic interpretability — `MechInterp.py`

**Key idea.** Capture GPT-2's internal state (the residual stream at layer 8) and decode it with a sparse autoencoder that was trained on millions of tokens. Each SAE feature tends to correspond to one human-interpretable concept.

**Runtime.** 1–2 minutes (longer on first run while it downloads ~1 GB).

**Expected output.** The 20 most active features, then **sparsity ≈ 99.8%**, then a plot (`mech.png`) showing which features fire at each token.

**Live demo.** Pick two or three feature numbers from the output and look them up at <https://www.neuronpedia.org/gpt2-small/res-jb>. Each has example texts that make it fire. This is the moment the idea clicks for most students. (The first feature or two are often tied to the start-of-text token and less interesting; skip to the others.)

**Exercise.** Change `text` to something about a single topic (cooking, code, sport) and see which features change.

**Questions to ask.**
- What is the difference between a *neuron* and an SAE *feature*? *(A neuron often responds to many unrelated things; SAE features are trained to be sparse so each tends to mean one thing.)*
- This tells us what GPT-2 represents. Does it tell us why it produced a particular output? *(Not by itself; that needs causal experiments such as turning features up or off.)*

### 4.8 Production LLM practice — `LMMExplain3.py` 3, 4, 8, 9

These need the API key (§ 2.3). Outputs below are from the test run; the wording will differ each time.

**3 — Chain-of-thought.** Both versions get **$4.25**. The step-by-step version shows its working: apples $2.00 + oranges $2.25.

*The point to make:* current Claude models reason internally even when asked for only the answer, so the "direct" answer isn't reasoning-free. What chain-of-thought adds is reasoning a human can read and check. Research shows written reasoning doesn't always reflect how the model actually computed the answer, so it's useful for auditing but not proof.

**4 — RAG with citations.** The retriever picks the two closest documents (scores ≈ 0.47 and 0.45) and Claude answers using only them, citing `[Source 1]` and `[Source 2]`. In the test run Claude also said the sources only partly answer the question and declined to fill the gap.

*The point to make:* citations show where the information came from, not how the model reasoned. Point out that refusing to go beyond the sources is the behaviour you want, and ask how you would check it at scale.

**8 — Observability.** Three prompts are traced with latency, token counts and simple risk flags. Expect "2 + 2" to be **low risk** (~1 s), the France GDP question **high risk** (specific dates and numbers), and the transformer explanation **medium** (it mentions a year, typically 2017, when transformers were introduced). Average latency was ≈ 7 s in the test run, mostly from the long transformer answer.

*The point to make:* these heuristics are crude. The transformer explanation is flagged for mentioning a correct date. Real systems use evaluation models or tools such as Arize Phoenix, LangSmith or Ragas. The structure (record every call, measure, flag, review) is what to take away.

**9 — Model card.** Writes and parses a Hugging Face model card. No API key needed.

*The point to make:* a model card is governance documentation. Walk through the sections: intended uses, out-of-scope uses, performance by dataset, known biases, ethical considerations. The numbers in this card are illustrative, not measured. Ask what evidence you'd need before you could write a real one.

### 4.9 TensorBoard — `tensorflow2.py`, `tensorflowExpanded.py`, `tensorboard1.py`

Despite the file names, the first two scripts use PyTorch. Each runs in a few seconds; then:

```bash
tensorboard --logdir runs
```

- `tensorflow2.py`: show the loss curve and the prediction of sin(π/2) climbing towards 1.0. Open the Graphs tab to show the model structure.
- `tensorflowExpanded.py`: a quick tour of each tab (Scalars, Images, Histograms, the Matplotlib figure, Text).
- `tensorboard1.py`: trains knowledge-graph embeddings on Hetionet (~4 minutes, environment C), then open the **Projector** tab. Search for a drug or disease and look at its nearest neighbours. This works well as an optional lab.

*The point to make:* watching training is the first explainability tool most teams use, and it catches problems (overfitting, dead layers, exploding weights) before any attribution method is needed.

---

## 5. Exercises

Ordered from easiest to hardest. Estimated times are for students who have run the example once.

1. **SHAP speed-up (15 min).** In the SHAP notebook, replace `shap.Explainer(model.predict, X_test)` with `shap.TreeExplainer(model)` and explain the whole test set. Compare run time and the feature ranking.
2. **LIME stability (20 min).** Run LIME on the cat image five times with `num_samples=100`, then five times with 1000. Describe how the masks change.
3. **Your own image (20 min).** Run LIME and Grad-CAM on a photo of your own. Do the two methods agree?
4. **Change ε (20 min).** In the adversarial notebook, try ε = 0.01, 0.05, 0.1 and 0.2. Plot adversarial accuracy against ε, and look at the images. At what point would a person notice?
5. **Fix the probe (45 min).** Make example 6 meaningful: add more sentences, and add a control where the labels are shuffled. Does accuracy still reach 100% at every layer?
6. **Sparser SAE (30 min).** In example 7, raise `sparsity_coeff` until fewer than 10% of features are active. What happens to reconstruction error?
7. **Feature hunting (45 min).** Using `MechInterp.py` and Neuronpedia, find one feature that fires for a concept you choose. Show three sentences where it's active and three where it isn't.
8. **Break the RAG (30 min, needs API key).** In example 4, ask a question the five documents can't answer. Does Claude say so, or does it make something up? Then add a misleading document and see whether it gets cited.
9. **Better hallucination flags (45 min, needs API key).** Improve `assess_hallucination_risk` in example 8 so the transformer explanation isn't flagged for a correct date. Discuss why heuristics like this are hard to get right.

---

## 6. Check-your-understanding questions

Answers are in brackets.

1. What is the difference between a global and a local explanation? Give one example of each from this course. *[Global: the overall SHAP bar chart. Local: the waterfall plot for one house, or LIME for one image.]*
2. SHAP and the random forest's own importance ranked features differently. Which is more reliable and why? *[SHAP, in general: it measures each feature's effect on predictions; impurity importance is biased toward features with many distinct values. Both are approximations.]*
3. Why does LIME give a different answer each time you run it? *[It samples random perturbations; with few samples, the fitted linear model varies.]*
4. Name one thing Grad-CAM can explain that LIME can't, and the reverse. *[Grad-CAM uses the network's internals, so it's fast and specific to a layer. LIME treats the model as a black box, so it works on any model, including non-CNNs.]*
5. A model has 97% accuracy. Why might that not be enough to deploy it? *[It may be fragile, as FGSM showed, or right for the wrong reasons.]*
6. In BERT's last layer, most tokens attend to the full stop. Does that mean the full stop is the most important word? *[No. Attention weights aren't measures of causal importance; punctuation often acts as a "sink" for attention.]*
7. A linear probe gets 100% accuracy at every layer. What else would you check before concluding anything? *[A control task with shuffled labels, more data, and a held-out test set; with few examples and many dimensions, a probe can memorise.]*
8. What problem do sparse autoencoders solve? *[Individual neurons respond to many unrelated concepts. SAEs re-express activations as a larger set of sparse features that each tend to mean one thing.]*
9. Does chain-of-thought show how an LLM actually reached its answer? *[Not necessarily. It shows reasoning the model wrote, which humans can check, but research shows it isn't always faithful to the underlying computation.]*
10. What does a RAG citation tell you, and what doesn't it? *[It tells you which source the answer draws on. It doesn't show the model's reasoning or guarantee the answer represents the source correctly.]*
11. Which technique would you use to explain a single decision of a gradient-boosted credit-scoring model to a regulator, and what caveat would you give? *[SHAP with TreeExplainer, giving the per-applicant waterfall; caveat that correlated features share credit and the explanation describes the model, not the real-world cause.]*

---

## 7. Troubleshooting in class

| Problem | Fix |
|---|---|
| `CUDA error: no kernel image is available` | Older GPU. Run with `CUDA_VISIBLE_DEVICES=` |
| `'HookedTransformer' was removed in TransformerLens 4.0` | Old copy of `MechInterp.py`; pull the latest version |
| `No module named 'pkg_resources'` (AmpliGraph) | `pip install "setuptools<81"` |
| `IProgress not found` from SHAP | `pip install ipywidgets` (harmless otherwise) |
| SHAP notebook taking 5+ minutes | `N_EXPLAIN` was set to `None`; set it back to 500 |
| TensorFlow prints CUDA/cuDNN errors at start-up | Harmless if no GPU is in use |
| `401 invalid x-api-key` | Key revoked, disabled or mistyped; create a new one |
| `BertModel LOAD REPORT … UNEXPECTED` | Harmless: the unused pre-training head is being skipped |
| Plot windows don't appear when running over SSH | Run with `MPLBACKEND=Agg`; the scripts save their figures to files anyway |
