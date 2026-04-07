"""
LLM & Generative AI Explainability - Python Examples
=====================================================
8 approaches covered:
  1. SHAP / LIME           - Feature attribution
  2. Chain-of-Thought      - Step-by-step reasoning
  3. RAG Transparency      - Source-grounded generation
  4. Attention Visualization - Token attention weights
  5. Probing Classifiers   - What's encoded in hidden states
  6. Sparse Autoencoders   - Mechanistic interpretability
  7. LLM Observability     - Production monitoring
  8. Model Cards           - Governance documentation

Install all dependencies:
  pip install anthropic shap transformers torch bertviz
              sentence-transformers scikit-learn numpy
              huggingface_hub evaluate datasets
"""
import os
from dotenv import load_dotenv
# ===========================================================
# 1. SHAP - Feature Attribution for LLM Predictions
# ===========================================================
# pip install shap transformers torch
def example_shap():
    import shap
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    def predict(texts):
        results = classifier(list(texts))
        return [1.0 if r["label"] == "POSITIVE" else 0.0 for r in results]

    explainer = shap.Explainer(predict, shap.maskers.Text(tokenizer=r"\W+"))

    text = "The movie was surprisingly good but a bit long."
    shap_values = explainer([text])

    print("=== SHAP Feature Attribution ===")
    print(f"Input: {text}")
    print(f"SHAP values (per token): {shap_values.values[0]}")
    print("Positive = pushed toward POSITIVE label")
    print("Negative = pushed toward NEGATIVE label")

    # In a Jupyter notebook, visualize with:
    # shap.plots.text(shap_values[0])


# ===========================================================
# 2. LIME - Local Interpretable Model-agnostic Explanations
# ===========================================================
# pip install lime transformers torch numpy
def example_lime():
    import numpy as np
    from lime.lime_text import LimeTextExplainer
    from transformers import pipeline

    classifier = pipeline(
        "text-classification",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    def predict_proba(texts):
        results = classifier(list(texts))
        probs = []
        for r in results:
            p = r["score"] if r["label"] == "POSITIVE" else 1 - r["score"]
            probs.append([1 - p, p])
        return np.array(probs)

    explainer = LimeTextExplainer(class_names=["NEGATIVE", "POSITIVE"])
    text = "The film had great acting but a terrible plot."

    explanation = explainer.explain_instance(
        text, predict_proba, num_features=6, num_samples=100
    )

    print("=== LIME Explanation ===")
    print(f"Input: {text}")
    print("Top features and their contribution:")
    for feature, weight in explanation.as_list():
        direction = "POSITIVE" if weight > 0 else "NEGATIVE"
        print(f"  '{feature}': {weight:+.4f} -> pushes toward {direction}")

    # In a Jupyter notebook, visualize with:
    # explanation.show_in_notebook()


# ===========================================================
# 3. Chain-of-Thought Prompting
# ===========================================================
# pip install anthropic
def example_chain_of_thought():
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDKEY"))  # uses ANTHROPIC_API_KEY env var

    question = (
        "A store sells apples for $0.50 each and oranges for $0.75 each. "
        "If Alice buys 4 apples and 3 oranges, how much does she spend in total?"
    )

    # Without CoT
    response_direct = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=128,
        messages=[{"role": "user", "content": question}]
    )

    # With Chain-of-Thought
    response_cot = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": question + "\n\nThink step by step before giving your final answer."
        }]
    )

    print("=== Chain-of-Thought ===")
    print(f"Question: {question}\n")
    print("Direct answer:")
    print(response_direct.content[0].text)
    print("\nChain-of-Thought answer:")
    print(response_cot.content[0].text)


# ===========================================================
# 4. RAG Transparency - Grounded, Citable Responses
# ===========================================================
# pip install anthropic sentence-transformers numpy
def example_rag():
    import numpy as np
    import anthropic
    from sentence_transformers import SentenceTransformer

    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDKEY"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    documents = [
        {"id": 0, "text": "LLMs use transformer architecture with self-attention mechanisms."},
        {"id": 1, "text": "SHAP values explain model predictions using cooperative game theory."},
        {"id": 2, "text": "Chain-of-thought prompting improves reasoning transparency in LLMs."},
        {"id": 3, "text": "Sparse autoencoders decompose model activations into monosemantic features."},
        {"id": 4, "text": "RAG grounds LLM outputs in retrieved documents, reducing hallucinations."},
    ]
    doc_texts = [d["text"] for d in documents]
    doc_embeddings = embedder.encode(doc_texts)

    def retrieve(query, top_k=2):
        query_emb = embedder.encode([query])
        scores = np.dot(doc_embeddings, query_emb.T).flatten()
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(documents[i], float(scores[i])) for i in top_idx]

    def rag_query(question):
        retrieved = retrieve(question)
        context = "\n".join([
            f"[Source {i+1}] {doc['text']}"
            for i, (doc, _) in enumerate(retrieved)
        ])
        prompt = (
            f"Answer using only the sources below. Cite each claim with [Source N].\n\n"
            f"Sources:\n{context}\n\nQuestion: {question}"
        )
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text, retrieved

    question = "How do LLMs achieve transparency in their outputs?"
    answer, sources = rag_query(question)

    print("=== RAG Transparency ===")
    print(f"Question: {question}\n")
    print(f"Answer:\n{answer}\n")
    print("Retrieved sources:")
    for doc, score in sources:
        print(f"  [score={score:.3f}] {doc['text']}")


# ===========================================================
# 5. Attention Visualization
# ===========================================================
# pip install transformers torch
def example_attention():
    import torch
    from transformers import AutoTokenizer, AutoModel

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    sentence = "The animal didn't cross the street because it was too tired."
    inputs = tokenizer(sentence, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    attention = outputs.attentions  # (num_layers, batch, heads, seq, seq)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    print("=== Attention Visualization ===")
    print(f"Sentence: {sentence}")
    print(f"Tokens: {tokens}\n")
    print(f"Model has {len(attention)} layers, {attention[0].shape[1]} heads each\n")

    # Analyse last layer, averaged across heads
    last_layer = attention[-1][0].mean(dim=0)  # (seq, seq)
    print("Top attention targets per token (last layer, avg heads):")
    for i, tok in enumerate(tokens):
        top_j = last_layer[i].argmax().item()
        score = last_layer[i][top_j].item()
        print(f"  '{tok}' -> attends most to '{tokens[top_j]}' (score={score:.3f})")

    # In a Jupyter notebook, use bertviz for interactive visualization:
    # from bertviz import head_view
    # head_view(attention, tokens)


# ===========================================================
# 6. Probing Classifiers - What's encoded in hidden states?
# ===========================================================
# pip install transformers torch scikit-learn
def example_probing():
    import torch
    import numpy as np
    from transformers import AutoTokenizer, AutoModel
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval()

    # Probe task: does the model encode noun vs verb at each layer?
    sentences = [
        ("The cat sat on the mat", "cat", "NOUN"),
        ("She runs every morning", "runs", "VERB"),
        ("The dog barked loudly", "dog", "NOUN"),
        ("He writes poetry daily", "writes", "VERB"),
        ("A bird flew overhead", "bird", "NOUN"),
        ("They play football here", "play", "VERB"),
        ("The tree fell down", "tree", "NOUN"),
        ("She dances gracefully", "dances", "VERB"),
        ("The book was heavy", "book", "NOUN"),
        ("He sings every night", "sings", "VERB"),
    ]

    def get_hidden_state(sentence, target_word, layer):
        inputs = tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        hidden = outputs.hidden_states[layer][0]
        toks = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        for i, tok in enumerate(toks):
            if target_word.lower() in tok:
                return hidden[i].numpy()
        return hidden[1].numpy()

    print("=== Probing Classifiers ===")
    print("Task: Can a linear probe detect NOUN vs VERB from hidden states?\n")

    y = np.array([1 if pos == "NOUN" else 0 for _, _, pos in sentences])

    for layer in [0, 3, 6, 9, 12]:
        X = np.array([get_hidden_state(s, w, layer) for s, w, _ in sentences])
        probe = LogisticRegression(max_iter=1000)
        scores = cross_val_score(probe, X, y, cv=3)
        print(f"  Layer {layer:2d}: accuracy = {scores.mean():.2%} (+/- {scores.std():.2%})")

    print("\nHigher accuracy = that layer encodes POS information more strongly.")


# ===========================================================
# 7. Sparse Autoencoder (Mechanistic Interpretability)
# ===========================================================
# pip install torch numpy
def example_sparse_autoencoder():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import numpy as np

    class SparseAutoencoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, sparsity_coeff=0.05):
            super().__init__()
            self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=False)
            self.sparsity_coeff = sparsity_coeff

        def forward(self, x):
            features = torch.relu(self.encoder(x))   # sparse activations
            reconstructed = self.decoder(features)
            return reconstructed, features

        def loss(self, x, reconstructed, features):
            recon_loss = ((x - reconstructed) ** 2).mean()
            sparsity_loss = features.abs().mean()  # L1 penalty
            return recon_loss + self.sparsity_coeff * sparsity_loss

    torch.manual_seed(42)
    input_dim, hidden_dim, n_samples = 64, 256, 500

    # Simulate LLM activations: 3 underlying concepts mixed together
    true_concepts = torch.randn(3, input_dim)
    activations = torch.relu(
        torch.randn(n_samples, 3) @ true_concepts
        + 0.1 * torch.randn(n_samples, input_dim)
    )

    sae = SparseAutoencoder(input_dim, hidden_dim)
    optimizer = optim.Adam(sae.parameters(), lr=1e-3)

    print("=== Sparse Autoencoder (Mechanistic Interpretability) ===")
    print(f"Input dim: {input_dim}, Dictionary size: {hidden_dim}\n")

    for epoch in range(600):
        reconstructed, features = sae(activations)
        loss = sae.loss(activations, reconstructed, features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 150 == 0:
            sparsity = (features > 0).float().mean().item()
            recon_err = ((activations - reconstructed) ** 2).mean().item()
            print(f"  Epoch {epoch+1}: total_loss={loss.item():.4f}, "
                  f"recon_err={recon_err:.4f}, "
                  f"active_features={sparsity:.1%}")

    with torch.no_grad():
        _, features = sae(activations)

    n_active = (features > 0).sum(dim=1).float().mean().item()
    print(f"\nAvg features active per sample: {n_active:.1f} / {hidden_dim} ({n_active/hidden_dim:.1%})")
    print("Low percentage = sparse = more interpretable, monosemantic features.")
    print("In practice, each active feature ideally encodes one human-interpretable concept.")


# ===========================================================
# 8. LLM Observability - Production Monitoring
# ===========================================================
# pip install anthropic
def example_observability():
    import re
    import time
    import anthropic
    from dataclasses import dataclass, field
    from datetime import datetime, timezone

    client = anthropic.Anthropic(api_key=os.environ.get("CLAUDKEY"))

    @dataclass
    class LLMTrace:
        timestamp: str
        prompt: str
        response: str
        latency_ms: float
        input_tokens: int
        output_tokens: int
        hallucination_risk: str = "unknown"
        flags: list = field(default_factory=list)

    def assess_hallucination_risk(response: str):
        """
        Heuristic checks. In production use a dedicated eval model
        or tools like Arize Phoenix, LangSmith, or Ragas.
        """
        flags = []
        if re.search(r"\b(19|20)\d{2}\b", response):
            flags.append("contains_specific_dates")
        if re.search(r"\$[\d,]+|\d+(\.\d+)?%", response):
            flags.append("contains_specific_numbers")
        if re.search(r"\b(always|never|guaranteed|certain)\b", response, re.I):
            flags.append("absolute_language")
        if re.search(r"\bI think\b.*\bcertainly\b", response, re.I):
            flags.append("conflicting_confidence")
        risk = "high" if len(flags) >= 2 else "medium" if flags else "low"
        return risk, flags

    def monitored_call(prompt: str) -> LLMTrace:
        start = time.time()
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": prompt}]
        )
        latency = (time.time() - start) * 1000
        text = response.content[0].text
        risk, flags = assess_hallucination_risk(text)
        return LLMTrace(
            timestamp=datetime.now(timezone.utc).isoformat(),
            prompt=prompt,
            response=text,
            latency_ms=round(latency, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            hallucination_risk=risk,
            flags=flags,
        )

    prompts = [
        "What is 2 + 2?",
        "What was the exact GDP of France in 2019 and what will it be in 2030?",
        "Explain what a transformer neural network is.",
    ]

    print("=== LLM Observability ===\n")
    traces = []
    for prompt in prompts:
        trace = monitored_call(prompt)
        traces.append(trace)
        print(f"Prompt : {prompt[:60]}")
        print(f"Latency: {trace.latency_ms}ms | "
              f"Tokens: {trace.input_tokens}in / {trace.output_tokens}out")
        print(f"Risk   : {trace.hallucination_risk} | Flags: {trace.flags or 'none'}")
        print(f"Response (truncated): {trace.response[:120]}...")
        print()

    avg_latency = sum(t.latency_ms for t in traces) / len(traces)
    high_risk = sum(1 for t in traces if t.hallucination_risk == "high")
    print(f"Summary — avg latency: {avg_latency:.0f}ms | "
          f"high-risk responses: {high_risk}/{len(traces)}")


# ===========================================================
# 9. Model Cards - Governance Documentation
# ===========================================================
# pip install huggingface_hub
def example_model_card():
    from huggingface_hub import ModelCard

    card_content = """\
---
language: en
license: apache-2.0
tags:
  - text-classification
  - explainable-ai
  - sentiment-analysis
---

# Sentiment Classifier v1

## Model description
DistilBERT fine-tuned for binary sentiment classification (positive / negative)
on the Stanford Sentiment Treebank (SST-2).

## Intended uses
- Product review analysis
- Customer feedback triage at scale

## Out-of-scope uses
- Medical or legal sentiment analysis
- Languages other than English
- High-stakes decisions without human review

## Performance

| Dataset         | Split | Accuracy | F1    |
|-----------------|-------|----------|-------|
| SST-2           | test  | 93.2%    | 93.1% |
| Amazon Reviews  | test  | 88.7%    | 88.5% |
| Yelp Polarity   | test  | 90.1%    | 90.0% |

## Limitations and known biases
- Trained primarily on movie reviews; may underperform on technical domains.
- Lower accuracy (~4% gap) detected on African American Vernacular English (AAVE).
- Struggles with sarcasm and irony.

## Fairness evaluation
Evaluated across writing styles, demographics, and topic domains.
Bias mitigation (re-sampling + adversarial training) is ongoing.

## Training data
Stanford Sentiment Treebank (SST-2): 67,349 labeled movie-review sentences.

## Ethical considerations
Do NOT use for screening job applications, loan approvals, or other
high-stakes decisions without additional validation and human oversight.
This model is subject to the EU AI Act transparency requirements for
high-risk AI systems.
"""

    card = ModelCard(card_content)
    print("=== Model Card ===\n")
    print(card.content)
    print("\n--- Card parsed successfully ---")
    # To publish: card.push_to_hub("my-org/sentiment-classifier-v1")


# ===========================================================
# Run all examples
# ===========================================================
if __name__ == "__main__":
    import sys

    examples = {
        "1": ("SHAP Feature Attribution",         example_shap),
        "2": ("LIME Explanation",                  example_lime),
        "3": ("Chain-of-Thought",                  example_chain_of_thought),
        "4": ("RAG Transparency",                  example_rag),
        "5": ("Attention Visualization",           example_attention),
        "6": ("Probing Classifiers",               example_probing),
        "7": ("Sparse Autoencoder",                example_sparse_autoencoder),
        "8": ("LLM Observability",                 example_observability),
        "9": ("Model Card",                        example_model_card),
    }

    def run_example(key):
        name, fn = examples[key]
        print(f"\n{'='*60}")
        print(f"Running example {key}: {name}")
        print(f"{'='*60}\n")
        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {name} failed: {e}")

    if len(sys.argv) > 1:
        key = sys.argv[1].strip().lower()
        if key == "all":
            print("Running all examples sequentially...\n")
            print("Note: examples 3, 4, 8 require ANTHROPIC_API_KEY to be set.\n")
            for k in examples:
                run_example(k)
                print()
        elif key in examples:
            run_example(key)
        else:
            print(f"Unknown example '{key}'.")
            print(f"Choose a number 1-9, or pass 'all' to run everything.")
            print()
            for k, (name, _) in examples.items():
                print(f"  {k}  {name}")
    else:
        print("LLM Explainability Examples")
        print("Usage: python llm_explainability_examples.py <number|all>")
        print()
        for k, (name, _) in examples.items():
            print(f"  {k}  {name}")
        print()
        print("Examples: python llm_explainability_examples.py 1")
        print("          python llm_explainability_examples.py all")
        print()
        print("Note: examples 3, 4, 8 require ANTHROPIC_API_KEY to be set.")
