# Explainability Examples — Quick Start

## 1. Set up (once)

Work from the repository folder. On this machine the environments already exist; on a new one, create them as in the [README](../README.md#setup).

| Environment | Activate | Runs |
|---|---|---|
| PyTorch | `source .venv-torch/bin/activate` | All `.py` scripts except `tensorboard1.py` |
| TensorFlow | Jupyter kernel **"Explainability (TensorFlow)"** | The three notebooks (`jupyter lab`) |
| AmpliGraph | `source .venv-kg/bin/activate` | `tensorboard1.py` only |

- **Claude API key** (needed for LLM examples 3, 4, 8): create a `.env` file next to `LMMExplain3.py` containing `ANTHROPIC_API_KEY=sk-ant-api03-...`. Never commit or show it.
- **Older NVIDIA GPU** (GTX 10-series): use the CUDA 12.6 PyTorch build (README § GPU), or add `CUDA_VISIBLE_DEVICES=` before a command to run on CPU.
- **VS Code:** "Python: Select Interpreter" → `.venv-torch/bin/python`.

## 2. Programs to run (suggested order)

| # | Run | Time | Main point |
|---|---|---|---|
| 1 | `1-ExplainabilitySHAPhousing.ipynb` | 1 min | SHAP and the forest's own importance agree on income, then disagree: SHAP ranks location far higher. Methods can disagree. |
| 2 | `2-ExplainabilityLimeCat.ipynb` | 10 s | Model is unsure (lynx 29%, plastic bag 15%). LIME shows which regions it used, and the mask changes run to run. |
| 3 | `python grad-cam.py` | 10 s | Top guess is "plastic bag" (9.9%), not the cat. Its heatmap (`heatmap.jpg`) lights up the petals; the "tabby" heatmap lights up the face. Check which class an explanation explains. Red = most support, blue = least. |
| 4 | `AdversarialMachineLearning.ipynb` | 20 s | Invisible pixel changes drop accuracy from ~97% to ~3%. Accurate is not the same as robust. |
| 5 | `python GNNExplainer.py` | 30 s | Explains a graph link prediction by the edges that matter (`explain.png`). It explains an overfitted model just as confidently. |
| 6 | `python LMMExplain3.py 1` and `2` | 15 s each | SHAP and LIME on text: "but" and "terrible" push sentiment negative. |
| 7 | `python LMMExplain3.py 5` | 10 s | BERT attention mostly lands on the full stop. Attention is not importance. |
| 8 | `python LMMExplain3.py 6` | 10 s | Probes score 100% at every layer, a warning that small probes can memorise. Needs controls. |
| 9 | `python LMMExplain3.py 7`, then `python MechInterp.py` | 5 s / 25 s | Sparse autoencoders turn activations into sparse, nameable features. Look up feature numbers at neuronpedia.org/gpt2-small/res-jb. |
| 10 | `python LMMExplain3.py 3` | 15 s | Chain-of-thought makes reasoning readable, but it may not be how the model actually computed the answer. |
| 11 | `python LMMExplain3.py 4` | 15 s | RAG with citations shows where an answer came from, not how it was reasoned. |
| 12 | `python LMMExplain3.py 8` | 25 s | Production tracing: latency, tokens, risk flags. The heuristics are crude; the habit of logging everything is the point. |
| 13 | `python LMMExplain3.py 9` | 1 s | A model card documents intended use, limits and bias for governance. |
| 14 | `python tensorflow2.py`, then `tensorboard --logdir runs` | 5 s | Watching training (loss, predictions, weights) is the first explainability tool most teams use. |

`python LMMExplain3.py` with no number lists all nine LLM examples; `python LMMExplain3.py all` runs them all.

## 3. Main points to take away

1. **Global vs local:** explain the model overall (SHAP bar chart) or one prediction (waterfall plot, LIME mask).
2. **An explanation is a model of the model.** It can be unstable (LIME), coarse (Grad-CAM) or misleading (attention).
3. **Methods disagree,** so use more than one and compare.
4. **Explanations don't measure quality.** They describe what a model does, even when it's overfitted, unsure or fragile.
5. **Looking inside** (probing, sparse autoencoders) shows what a model represents, not why it made a decision.
6. **In production,** auditability comes from citations, tracing and documentation, not from trusting the model's own account.
