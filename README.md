# Prompt Engineering at Scale: Zero-Shot, Few-Shot & Chain-of-Thought Across Qwen3.5 Model Sizes

> **TL;DR**: A systematic 12-experiment study (3 model sizes × 4 prompting strategies) measuring how prompt design and model scale affect an LLM's ability to classify political answer clarity — without any fine-tuning. Includes subgroup analysis (does input length matter?) and full error/confusion analysis for the best configuration.

---

## 🔍 Research Question

Given the same task — classifying a politician's answer as *Clear Reply*, *Ambivalent*, or *Clear Non-Reply* — **how much does prompting strategy matter compared to model size**, when using off-the-shelf LLMs with zero parameter updates?

This project runs a full grid search over:

| Axis | Values |
|---|---|
| **Model size** | Qwen3.5-0.8B, Qwen3.5-2B, Qwen3.5-4B |
| **Prompting strategy** | Zero-Shot, Few-Shot (6 examples), Chain-of-Thought (CoT), Few-Shot + CoT |

→ **12 model × strategy combinations**, each evaluated on the full 690-example validation set.

---

## 📊 Dataset

- **Source**: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) (HuggingFace Hub)
- Stratified 80/20 train/validation split (seed=42); 308-example unlabeled test set for final submission
- Class distribution is imbalanced: Ambivalent ≈ 59–67% across splits, which motivates **Macro F1** as the primary metric (not accuracy)
- Length features (question/answer word counts) are computed and used to bucket examples into short/long subgroups for later analysis

---

## 🧪 Prompting Strategies Implemented

All four prompts follow a consistent 4-component structure (instruction + label definitions + input + output format), with a shared system message defining the LLM's role:

| Strategy | Description |
|---|---|
| **Zero-Shot** | Label definitions + the question/answer pair, with a strict output-format constraint |
| **Few-Shot** | Adds 6 balanced labeled examples (2 per class, fixed seed) before the target pair |
| **Chain-of-Thought (CoT)** | Asks the model to reason in 2–3 sentences before outputting a `Final Label:` line |
| **Few-Shot + CoT** | Combines 3 worked examples (each with reasoning + final label) with the CoT output format |

A **cascaded output parser** robustly extracts the predicted label from raw generations — handling `Final Label:` lines, label variants/casing, Qwen3.5 `<think>...</think>` blocks, and falling back to the majority class (`Ambivalent`) if nothing matches.

---

## ⚙️ Experimental Setup

- **Models**: Qwen3.5-0.8B / 2B / 4B, loaded one at a time in float16 with `device_map="auto"` to manage VRAM
- **Decoding**: Greedy (`do_sample=False`) for full reproducibility, seed fixed at 42
- **Thinking mode disabled** (`enable_thinking=False`) for direct, parseable label outputs
- **Token budget**: 100 new tokens for direct strategies, 400 for CoT strategies (to leave room for reasoning)
- Results saved incrementally to CSV after each strategy as a safeguard against long-running interruptions

---

> 💡 The full table prints in **Cell 13 (Results Analysis)**. Four plots are also generated:
> - `macro_f1_comparison.png` — grouped bar chart, all 12 combinations
> - `model_size_effect.png` — scaling curve (does bigger = better?)
> - `heatmap_macro_f1.png` — heatmap of model × strategy
> - `perclass_f1_4b.png` — per-class F1 for the best configuration

### Analysis Included

- **Subgroup analysis**: Macro F1 broken down by question/answer length (short vs. long, split on training-set median) for the best system — reveals whether longer inputs are harder to classify
- **Error analysis**: confusion matrix, per-class error rates, most common misclassification pairs, and concrete worked examples of failure cases for the best system (Qwen3.5-4B, Zero-Shot)

---

## 🛠️ Tech Stack

- **Python**, **PyTorch**, **HuggingFace Transformers** (dev build, for Qwen3.5 support) & **Datasets**
- **Scikit-learn** (metrics, confusion matrices, classification reports)
- **Matplotlib / Seaborn** (visualizations)
- Run on Kaggle's GPU environment (sequential model loading to manage VRAM across 0.8B–4B models)

---

## 📁 Repository Structure

```
prompt-engineering-llm-scaling/
├── README.md
├── notebook.ipynb               # Full 12-experiment grid + analysis
├── requirements.txt
└── results/
    ├── macro_f1_comparison.png
    ├── model_size_effect.png
    ├── heatmap_macro_f1.png
    └── perclass_f1_4b.png
```

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
```

**Notes:**
- The first cell installs a development build of `transformers` (required for `qwen3_5` model support) — **restart the kernel afterward**.
- The full 12-combination grid on 690 validation examples is GPU-intensive and long-running; results are checkpointed to CSV after each strategy so the run can be resumed if interrupted.
- Requires GPU access (designed for Kaggle's GPU environment).

---

## 🔗 Related Projects

This study is part of a 3-phase exploration of detecting evasive answers in political interviews:

1. [**BERT Fine-Tuning From Scratch**](#) — a custom PyTorch training loop establishing a fine-tuned baseline
2. **Prompt Engineering at Scale** *(this repo)* — testing how far prompting alone (no fine-tuning) can go
3. [**D3-Agentic: Multi-Agent LLM Reasoning**](#) — building on these findings with a 4-agent reasoning pipeline and comparing all approaches head-to-head

---

## 🙏 Acknowledgments

Dataset: [`ailsntua/QEvasion`](https://huggingface.co/datasets/ailsntua/QEvasion) on HuggingFace Hub. Developed as part of NLP/LLM coursework, focused on rigorous, reproducible prompt-engineering experimentation.
