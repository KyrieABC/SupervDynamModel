# Supervised Dynamics Model (State-Action → Next-State)

A modular, production-style PyTorch project for learning system dynamics:

$$f_\theta(s_t, a_t) \rightarrow s_{t+1}$$

---

# 🚀 Overview

This project learns a **forward dynamics model**, a core component in:

* Model-Based Reinforcement Learning (MBRL)
* Robotics and control systems
* System identification
* World models / simulation learning

The goal is to approximate how a system evolves over time:

$$s_{t+1} = f_\theta(s_t, a_t)$$

## ⚙️ Technical Stack

### Core
* Python 3.10+
* PyTorch (model + training)
* NumPy (simulation + data)
* scikit-learn (normalization + dataset split)

### Visualization
* Matplotlib (plots + trajectories)
* TensorBoard (training logs)

### Engineering
* argparse (CLI interface)
* JSON configs (experiment control)
* src/ layout (modular packaging)

---

# 🧠 System Dynamics

### State Definition
$$s_t = [x_t, y_t, v_{x,t}, v_{y,t}]$$

### Action Definition
$$a_t = [a_{x,t}, a_{y,t}]$$

### Transition Function (Physics)
$$v_{x,t+1} = v_{x,t} + (a_{x,t} - \lambda v_{x,t}) \cdot dt$$
$$v_{y,t+1} = v_{y,t} + (a_{y,t} - \lambda v_{y,t}) \cdot dt$$
$$x_{t+1} = x_t + v_{x,t+1} \cdot dt$$
$$y_{t+1} = y_t + v_{y,t+1} \cdot dt$$

Where:
* $\lambda$ = damping coefficient
* $dt$ = timestep

---

# 🧠 Learning Objective

We train a neural network:

$$\hat{s}_{t+1} = f_\theta([s_t, a_t])$$

### Loss Function
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \| s_{t+1}^{(i)} - \hat{s}_{t+1}^{(i)} \|^2$$

---

# 🏗️ Model Architecture

### Baseline: MLP
* **Input:** state_dim + action_dim
* **Hidden:** [128 → 128 → 64]
* **Output:** state_dim
* **Activation:** ReLU
* **Initialization:** Kaiming Uniform

---

# 🔄 Full Workflow

### Step 1: Data Generation
Simulator → Random Actions → Transitions $(s_t, a_t, s_{t+1})$

Stored as:
* states.npy
* actions.npy
* next_states.npy

### Step 2: Train / Validation Split
$$\mathcal{D} = \mathcal{D}_{train} \cup \mathcal{D}_{val}$$

### Step 3: Normalization
Let $x = [s_t, a_t]$

$$x_{scaled} = \frac{x - \mu_x}{\sigma_x}$$
$$y_{scaled} = \frac{s_{t+1} - \mu_y}{\sigma_y}$$

### Step 4: Training
Forward pass: $\hat{y} = f_\theta(x)$

Loss: $\mathcal{L} = \|y - \hat{y}\|^2$

Optimization: $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$

### Step 5: Evaluation
**One-step prediction:**
$$s_{t+1} \approx \hat{s}_{t+1}$$

**Multi-step rollout:**

$$
\hat{s}_{t+k} = f_\theta(\hat{s}_{t+k-1}, a_{t+k-1})
$$

---

# 📊 Metrics

* **MSE** $= \frac{1}{N} \sum (y - \hat{y})^2$
* **RMSE** $= \sqrt{MSE}$
* **MAE** $= \frac{1}{N} \sum |y - \hat{y}|$

---

# 📁 Project Structure

```text
configs/          experiment configs
data/             dataset generation
scripts/          CLI scripts
src/
  simulation/     physics engine
  data/           dataset + preprocessing
  models/         neural networks
  training/       training loop
  evaluation/     metrics + rollout
  utils/          helpers
```

---

# ▶️ How to Run

Follow these steps to run the full pipeline:

---

## 1. Clone the repository

```bash
git clone <your-repo-url>
cd dynamics-model-repo
```
## 2. Create a virtual environment
```macOS/Linux
python -m venv .venv
source .venv/bin/activate
```
```Windows
python -m venv .venv
.venv\Scripts\Activate.ps1
```
## 3. Install dependencies
```
pip install -r requirements.txt
pip install -e .
```
## 4. Generate Synthetic dataset
```
python data/generate_dataset.py --config configs/default.json
```
## 5. Train the Model
```
python scripts/train.py --config configs/default.json
```
### Outputs:
```
artifacts/runs/default_run/
├── checkpoints/
│   ├── best.pt
│   └── last.pt
├── loss_curve.png
├── scaler.pkl
├── train_config_snapshot.json
```
## 6. Evaluate the Model
```
python scripts/evaluate.py --config configs/default.json
```
### Outputs:
```
artifacts/eval/default_run/
├── metrics.json
├── rollout_xy.png
├── component_scatter.png
├── state_component_timeseries.png
```

---

# 📈 Example Results

* **One-step prediction:** Low MSE → good immediate dynamics approximation.
* **Multi-step rollout:** Tests stability of learned model; reveals compounding errors.

---

# 🔬 Key Insight

Even if one-step loss is low:

$$
\hat{s}_{t+1} \approx s_{t+1}
$$

multi-step rollout can diverge:

$$
\hat{s}_{t+k} \neq s_{t+k}
$$

---

# 🚀 Future Improvements

* **Predict state delta:** $\Delta s_t = s_{t+1} - s_t$
* **Probabilistic dynamics model**
* **Ensemble models**
* **Uncertainty-aware rollouts**
* **Longer horizon training objectives**
