# 🤝 Contributing to DeepBench

First off, thank you for considering contributing to DeepBench! It's people like you that make the open-source community such an amazing place to learn, inspire, and create.

We believe that **AI research tools should be accessible**, and we welcome contributions from everyone—whether you're a seasoned AI researcher or a student making your first Pull Request.

---

## 🗺️ What Can I Contribute?

We are actively looking for help with the following features. If you want to work on one, please check the Issues tab to ensure no one else is already working on it!

### 🟢 Beginner Friendly
* **Documentation:** Fix typos in the README, add comments to code, or improve the installation guide.
* **UI Tweaks:** Improve the Streamlit CSS, add tooltips to buttons, or make the graphs more interactive.
* **New Benchmarks:** Add simple metric calculations (like BLEU or ROUGE) to `benchmarks.py`.

### 🟡 Intermediate
* **[ ] GGUF Support:** Implement model loading for `.gguf` files using the `ctransformers` library (great for CPU users!).
* **[ ] Weights & Biases Logging:** Add an option to log experiment results to a W&B dashboard.
* **[ ] Enhanced Diagnostics:** Add a "Layer Visualizer" that shows the actual weights distribution of a selected layer.

### 🔴 Advanced (The "Hard" Stuff)
* **[ ] FlashAttention-2:** Add a toggle to enable `use_flash_attention_2=True` in the `ModelManager` class for supported GPUs.
* **[ ] Custom Dataset Upload:** Allow users to upload their own `.txt` or `.csv` files for perplexity testing in the Battle Arena.

---

## 📏 Contribution Rules

1.  **Code Style:** We use standard Python conventions (PEP 8). Please ensure your variable names are descriptive (e.g., `model_id` instead of `m`).
2.  **No Breaking Changes:** If you change a core function in `backend.py`, make sure the "Battle Arena" and "Playground" tabs still work.
3.  **One Feature, One PR:** Please do not bundle multiple unrelated features into a single Pull Request.
4.  **Test It:** Before submitting, run the app locally (`streamlit run app.py`) to make sure it doesn't crash.

---

## 🚀 Step-by-Step Guide for Newbies

Never made an open-source contribution before? No problem! Follow these steps:

### Step 1: Fork the Repository
Click the **Fork** button in the top-right corner of this page. This creates a copy of the code under your own GitHub account.

### Step 2: Clone Your Fork
Open your terminal (Command Prompt/Terminal) and run:
```bash
# Replace 'your-username' with your actual GitHub username
git clone [https://github.com/sumitrwk90/deepbench.git](https://github.com/sumitrwk90/deepbench.git)
cd deepbench
```
### Step 3: Create a Branch
Never work on the main branch directly. Create a new branch for your specific feature:
```bash
# Example: git checkout -b add-flash-attention
git checkout -b your-feature-name
```
### Step 4: Set Up Environment
Install the requirements to make sure you can run the code:
```bash
pip install -r requirements.txt
```
### Step 5: Make Your Changes
Open the code in your favorite editor (VS Code, PyCharm, etc.) and write your awesome code!

Tip: Test your changes frequently by running:
```bash
streamlit run app.py
```
### Step 6: Commit and Push
Once you are happy with your work, save it to your branch:
```bash
git add .
git commit -m "Added support for FlashAttention-2"
git push origin your-feature-name
```
### Step 7: Create a Pull Request (PR)
Go to your forked repository on GitHub.

You will see a banner saying "Compare & pull request". Click it!

Write a clear title and description of what you made changes.

Click "Create pull request".

And Done...

🎉 Congratulations! You've just submitted your contribution. Me being maintainer will review your code, give feedback, and merge it into the project.

## THANK YOU...