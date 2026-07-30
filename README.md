# 🌟 CosAdam: Hyperparameter-Robust Adaptive Optimization via Cosine-Guided Trajectory Stabilization

**A computationally efficient, theoretically grounded deep learning optimizer for vision and language models, featuring directional consistency for variance suppression and stable non-convex convergence.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YourUsername/CosAdam/blob/main/colab_setup.ipynb)

---

## 📝 Abstract

🎯 **Objectives:** To develop and validate CosAdam, an optimization algorithm that modulates adaptive learning rates using a Directional Consistency Factor (DCF), and to rigorously evaluate its trajectory stabilization, hyperparameter robustness, and computational efficiency against standard baselines.

🔬 **Methods:** The optimizer was evaluated across two distinct modalities: computer vision (training DeiT-Small from scratch on CIFAR-100) and natural language processing (fine-tuning BERT-base and DistilBERT on AG News). CosAdam was benchmarked against AdamW, Lion, and SAM. Performance and robustness were comprehensively assessed via multi-seed evaluation, structural ablation on hyperparameters ($c$ and $\alpha_{\text{cos}}$), and rigorous statistical testing with Benjamini-Hochberg FDR corrections.

📊 **Results:** 
- 👁️ **Vision (DeiT-Small):** CosAdam achieved 86.57% accuracy, establishing statistical parity with AdamW while mitigating the extreme trajectory volatility observed in Lion. Compared to SAM, it delivered highly competitive generalization at a fraction (1/7th) of the computational cost.
- 🗣️ **Language (BERT-base):** CosAdam achieved 94.07% accuracy, perfectly matching the generalization of both AdamW and SAM. It bypassed SAM's severe 4.3× training time penalty.
- 🛡️ **Robustness:** Ablation studies revealed a remarkably flat hyperparameter-response surface. Accuracy varied by less than 0.2% across an order of magnitude of hyperparameter combinations, eliminating the need for exhaustive grid searches.
- 📉 **Stability:** CosAdam reduced run-to-run variance, achieving a Coefficient of Variation (CV) as low as 0.09%, establishing highly reproducible optimization trajectories.

---

## 🚀 Key Features

- 🧭 **Cosine-Guided Stabilization** – Utilizes a monotonic second-moment tracking integrated with a smoothed cosine similarity factor to dynamically penalize destructive gradient oscillations.
- 📐 **Theoretical Convergence** – Backed by a formal proof (Theorem 1) guaranteeing convergence bounded by a stable variance neighborhood in non-convex landscapes.
- ⚖️ **Pareto-Optimal Compute** – Achieves generalization comparable to sharpness-aware methods (SAM) with only a ~14% computational overhead relative to standard AdamW, maintaining $\mathcal{O}(N)$ complexity.
- 🧩 **Plug-and-Play Robustness** – Flat hyperparameter-response surface drastically reduces the necessity for extensive grid search tuning.
- 📈 **Rigorous Statistical Validation** – All comparative claims are backed by multiple random seeds, Benjamini-Hochberg FDR-adjusted $p$-values, bootstrapped 95% CIs, and Cohen's $d_z$ effect sizes.

---

## 🗂️ Dataset & Architecture Summary

| Modality | Dataset | Backbone Architecture | Parameters | Input Specification |
|----------|---------|-----------------------|------------|---------------------|
| 🖼️ **Vision** | CIFAR-100 | DeiT-Small (12 Blocks, 6 Heads) | ~22.0M | 224 × 224 × 3 Image |
| 📖 **Language** | AG News | BERT-base (12 Layers, 12 Heads) | ~110.0M | Tokenized Text (Max 128) |
| 📖 **Language** (Ablation)| AG News | DistilBERT (6 Layers, 12 Heads) | ~66.0M | Tokenized Text (Max 128) |

---

## 🏆 Performance Highlights

| Metric | CosAdam (Vision / DeiT) | CosAdam (NLP / BERT) |
|--------|-------------------------|----------------------|
| 🎯 **Accuracy (Mean $\pm$ SD)** | 86.57% $\pm$ 0.11% | 94.07% $\pm$ 0.07% |
| ⚓ **Stability (CV%)** | 0.12% (Config: $c=0.5, \alpha=0.9$) | 0.09% (Config: $c=0.5, \alpha=0.9$) |
| ⏱️ **Epoch Time vs SAM** | 186.06s (vs SAM: 1311.31s) | 737.32s (vs SAM: 3169.52s) |

> 💡 *Note: Results are validated over multiple independent seeds (4 for Vision, 3 for NLP). Differences against AdamW are statistically non-significant (ns), proving parity without extra tuning.*

---

## ⚙️ Implementation Details

All experiments are implemented in **PyTorch 2.0+** utilizing robust standard libraries for transformers and mixed-precision computing.

### 🎛️ Training Hyperparameters & Configuration
| Component | Specification |
|-----------|---------------|
| 📚 **Framework** | PyTorch 2.0+, torcheval, HuggingFace `transformers` |
| 🧮 **Base Optimizer Params**| $\beta_1=0.9$, $\beta_2=0.999$, $\epsilon=10^{-8}$, decoupled weight decay |
| 🔑 **CosAdam Specifics** | Scaling factor $c=0.5$, EMA momentum $\alpha_{\text{cos}}=0.9$ |
| 📦 **Batch Size** | 64 (DeiT/BERT), 128 (DistilBERT) |
| 📉 **LR Schedule** | Linear Warm-up (2 epochs Vision / 10% steps NLP) + Cosine Annealing |
| ⚡ **Precision** | Automatic Mixed Precision (AMP) |
| 🔬 **Statistical Rigor** | Paired two-tailed $t$-tests, BH-FDR correction, BCa Bootstrapping |

---

## 💻 Environment & Hardware

### 🖥️ System Requirements
- **OS:** Linux (Ubuntu 20.04+), Windows 10/11
- **Python:** 3.8 – 3.10  
- **CUDA:** 11.7 or 11.8 (required for GPU training)  
- **GPU:** NVIDIA RTX 3060 (12 GB VRAM) minimum; RTX 3090 / A100 recommended for full reproduction  
- **RAM:** 32 GB or higher  

### 🛠️ Software Dependencies
All required packages are listed in `requirements.txt`. Core libraries include:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `transformers>=4.30.0` (for BERT/DistilBERT)
- `timm>=0.9.0` (for DeiT models)
- `scipy`, `scikit-learn`, `pandas`, `numpy` (for statistical testing & metrics)
- `matplotlib`, `seaborn` (for visualization)

---

## 📥 Installation

Clone the repository and install dependencies:

```bash
git clone [https://github.com/YourUsername/CosAdam.git](https://github.com/YourUsername/CosAdam.git)
cd CosAdam
pip install -r requirements.txt
