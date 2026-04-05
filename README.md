# Motor Insurance Premium Pricing Engine 🚗💡

Welcome to the **Neuro-Fuzzy Motor Insurance Premium Pricing Engine** repository. This repository hosts a state-of-the-art hybrid machine learning framework developed to evaluate motor insurance risks, accurately calculate pure premium pricing, and ensure systemic financial solvency.

Traditionally, insurers use Generalized Linear Models (GLMs), which rely on rigid mathematical boundaries. This project pushes beyond classical actuarial limits by integrating an advanced Adaptive Neuro-Fuzzy Inference System (ANFIS) and a Deep Multi-Layer Perceptron (MLP) Stacker. The result is a mathematically rigorous, interpretably fluid, and financially solvent engine tailored for immediate modern underwriting operations.

---

## 🚀 Key Features

* **Neuro-Fuzzy Logic (ANFIS):** Translates complex, non-linear data (like age, vehicle value, etc.) into continuous, overlapping risk categories using dynamically optimized Gaussian membership functions built via PyTorch.
* **Deep MLP Meta-Learner:** An advanced Stacking Ensemble that seamlessly blends out-of-fold predictions from a benchmark GLM and the ANFIS model to generate highly calibrated pure premium quotes.
* **Strict Actuarial Solvency:** Features an Isotonic Regression layer to firmly calibrate outputs to real-world claim costs—achieving a structural 1.015 Predicted-to-Observed loss ratio (ensuring a 1.5% foundational profit margin buffer).
* **Supercharged Risk Segmentation:** Radically outperforms traditional linear bounds. Reduces MAE heavily and elevates risk differentiation ability (yielding a Gini Index improvement up to 0.1689). 
* **Interactive Underwriter Web App:** Breaks out of the terminal. Directly supplies underwriters with an elegant GUI that instantly loads interactive metrics, risk tiers, and continuous probabilistic time-to-claim survival curves.

---

## 🏗️ Technical Architecture

1. **Pre-Processing Engine:** Encodes standard policyholder inputs and dynamically models risk features.
2. **GLM Baseline Pipeline:** Traditional Poisson/Gamma models computing isolated frequency and severity.
3. **PyTorch ANFIS Component:** Data-driven fuzzification learning complex, intersectional risks that linear methods miss.
4. **Deep MLP Stacker:** Blends the GLM and ANFIS signals.
5. **Calibrator:** Isotonic regression enforcing financial safety.
6. **Deploy / GUI inference:** Models are instantly dumped to `saved_models.joblib` and inferred via the integrated real-time Web App interface.

---

## 📁 Repository Structure

```text
├── webapp/                      # Interactive Web App UI and backend integration
│   ├── index.html               # Frontend UI built with an elegant, modern dark design
│   └── inference.py             # Inference pipeline serving model predictions to the API
├── premium_app_script.py        # Core model training, fuzzification pipeline, and logic
├── app.py                       # Main application runner configuration
├── saved_models.joblib          # Persisted ML models (GLM, PyTorch ANFIS, Stacker, Isotonic Calibrator)
├── thesis_*.md                  # Extensive academic markdown chapters governing the math and methodology
└── *.png                        # Visual assets (Lorenz Curves, Decile Lift, Survival curves)
```

---

## 📈 Visualizing Core Results

The thesis documentation (`thesis_results.md`, `thesis_conclusion.md`) contains the breakdown of the performance metrics. Key visual validations produced by this engine include:
- **Membership Function Optimization (`mf_plots.png`)** proving continuous, fuzzy risk mapping.
- **Lorenz Curves & Decile Lift (`lorenz_curves.png`, `decile_lift.png`)** verifying top-tier risk segmentation of the worst 10% of drivers.
- **Probabilistic Degradation (`survival_curves.png`)** demonstrating the varying exponential temporal risk over a year.

---

## 💻 Getting Started

### 1. Requirements
Ensure you have Python 3.10+ installed.

```bash
# Clone the repository
git clone https://github.com/your-username/motor-insurance-premium-web-app.git
cd motor-insurance-premium-web-app

# Standard Python ML dependencies required 
pip install -r requirements.txt 
# (Includes PyTorch, Scikit-Learn, Pandas, Numpy, Flask/FastAPI depending on backend)
```

### 2. Training the Underwriting Engine
If you need to retrain the models (GLM baseline, ANFIS PyTorch model, and MLP Stacker) from scratch:
```bash
python premium_app_script.py
```
*(Note: A highly calibrated production version is already serialized into `saved_models.joblib`)*

### 3. Launching the Underwriter App
The system is designed to run easily straight out of the box. Once you've downloaded the repository:

1. **Start the Backend:** First, spin up the backend inference server:
```bash
python webapp/backend.py
```
*(Check your exact backend filename if using app.py instead).*

2. **Start the Frontend:** Next, simply open `webapp/index.html` in any modern web browser or run it via a local live server to interact with the UI.


## 📜 License
This software is provided for academic, research, and non-commercial actuarial development.
