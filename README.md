# 🧪 DeepBench: AI Researcher Workbench

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange?style=for-the-badge&logo=pytorch)
![HuggingFace](https://img.shields.io/badge/Hugging%20Face-Hub-yellow?style=for-the-badge&logo=huggingface)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **"Don't just train models. Dissect them."**

## 🚀 Overview

**DeepBench** is a unified, end-to-end research dashboard designed to bridge the gap between theoretical architecture and practical deployment. It replaces scattered scripts and notebooks with a **single, "Battle Arena" interface**.

Whether you are deciding between an **A100 vs. RTX 4090**, comparing **FP16 vs. 8-bit Quantization**, or performing **Virtual Surgery (Ablation)** on a live neural network, DeepBench gives you the insights instantly.

---

## 💎 Key Capabilities

### ⚔️ The Battle Arena (Side-by-Side Comparison)
* **Dual-Model Loading:** Load two different models (e.g., `Llama-2` vs `Mistral`) simultaneously.
* **Real-Time Benchmarking:** Run simulated or real metrics (Perplexity, MMLU, GSM8K, ARC) on both models instantly.
* **Quantization Toggle:** Compare a "Raw" FP16 model against its **8-bit Quantized** version to measure performance degradation.
* **Visual Radar Charts:** Automatically generates spider graphs to visualize trade-offs (e.g., *Model A is better at Math, Model B is better at Logic*).

### ✂️ Ablation Lab (Virtual Surgery)
* **"No-Code" Ablation:** Systematically disable or corrupt specific layers **without rewriting model code** using PyTorch Hooks.
* **NetworkX Visualization:** View interactive, node-link diagrams of the model's internal graph.
* **Sensitivity Heatmaps:** Visualize which layers contribute most to the model's output integrity.

### 💾 Hardware Forecast & Diagnostics
* **VRAM Estimator:** "Will it fit?" Calculator. Instantly checks if a model fits on your GPU for Training (FP32), Inference (FP16), or Quantized (Int8).
* **Model X-Ray:** Inspect the raw PyTorch layer structure (`Linear`, `Attention`, `LayerNorm`) inside the dashboard.

---

## 🛠️ Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Frontend** | `Streamlit` | Artistic, dark-mode UI with Tabs and Sidebar control. |
| **Core AI** | `PyTorch` + `Transformers` | Model loading, inference, and hooks. |
| **Optimization** | `BitsAndBytes` | Real-time 8-bit quantization logic. |
| **Visualization** | `Plotly` + `NetworkX` | Interactive radar charts and graph theory plotting. |
| **Data** | `Hugging Face Hub` | Live API connection to fetch 500k+ models. |

---

## ⚡ Quick Start

### 1. Clone & Setup
```bash
git clone [https://github.com/sumitrwk90/deepbench.git](https://github.com/sumitrwk90/deepbench.git)
cd deepbench

# Create a virtual environment (Recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

```bash
# Install required packages
pip install -r requirements.txt

# Run app
streamlit run app.py
```