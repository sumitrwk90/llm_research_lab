# 🧠 Neural Architect: Model Insight Lab

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-ff4b4b)
![HuggingFace](https://img.shields.io/badge/AI-Hugging%20Face-yellow)
![PyTorch](https://img.shields.io/badge/Backend-PyTorch-orange)

> **"Before creating a new architecture, we must deeply understand the performance of existing ones."**

## 📖 Project Overview

**Neural Architect** is an end-to-end AI research dashboard designed to bridge the gap between theoretical architecture design and practical model performance. 

Instead of blindly building new neural networks, this tool allows researchers to **dynamically search, load, and benchmark** existing Large Language Models (LLMs) from the Hugging Face Hub in real-time. It specifically focuses on enabling comparative analysis between different architectural paradigms (e.g., standard **Attention/Transformers** vs. emerging **Recurrent/RNN/State-Space** models) of similar parameter sizes.

### 🌟 Key Features
* **Automated Model Discovery:** Connects directly to the Hugging Face API to fetch live models based on architecture tags and popularity.
* **Dynamic Benchmarking:** Instantly loads models into memory to calculate real-world metrics like **Perplexity** (using the WikiText dataset).
* **Interactive Visualization:** Renders artistic, high-contrast plots using Plotly to visualize performance tradeoffs.
* **Modular Architecture:** Clean separation between the calculation engine (`backend.py`) and the presentation layer (`app.py`).

---

## 🛠️ Tech Stack & Skills Demonstrated

* **Frontend:** Streamlit (Custom CSS styling, State management).
* **AI/ML Core:** PyTorch, Hugging Face Transformers, Accelerate.
* **Data Handling:** Pandas, Hugging Face Datasets.
* **Visualization:** Plotly Express.
* **API Integration:** `huggingface_hub` API for metadata filtering.

---

## 🚀 How to Run the Project

Follow these steps to deploy the research lab on your local machine.

### 1. Prerequisites
Ensure you have **Python 3.8+** installed. A GPU (CUDA) is recommended for faster inference but not required (the code automatically falls back to CPU).

### 2. Installation
Clone the repository (or download the files) and install the dependencies:

```bash
# Install required packages
pip install -r requirements.txt

# Run app
streamlit run app.py