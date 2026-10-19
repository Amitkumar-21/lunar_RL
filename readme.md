# LunarLander-v3 Deep Reinforcement Learning (PPO)

A reinforcement learning project implementing, benchmarking, and optimizing Proximal Policy Optimization (PPO) agents on Farama Foundation's **`LunarLander-v3`** environment using Stable-Baselines3 and PyTorch.

---

## 📌 Project Overview

The objective of **LunarLander-v3** is to pilot a lunar lander from the top of the screen to safely touch down on a designated landing pad marked by two flags. 

- **State Space (8 dimensions):** Horizontal & vertical coordinates $(x, y)$, linear velocities $(v_x, v_y)$, lander angle $\theta$, angular velocity $\omega$, and two boolean flags indicating whether each lander leg is in ground contact.
- **Action Space (Discrete 4):**
  - `0`: Do nothing
  - `1`: Fire left orientation engine
  - `2`: Fire main engine
  - `3`: Fire right orientation engine
- **Reward Dynamics:** Moving toward the pad yields positive reward; moving away is penalized; landing safely awards $+100$; crashing penalizes $-100$; firing the main engine costs $-0.30$ per frame; firing side engines costs $-0.03$ per frame.
- **Success Benchmark Threshold:** An episode is considered **successful** if its total return is **$\ge 200$**. In RL literature, the environment is officially considered **solved** when the agent consistently achieves an average score $\ge 200$ across randomized seeds.

---

## ⚙️ Hyperparameter Configurations

This project benchmarks default PPO against a tuned configuration designed to prevent entropy collapse and stabilize long-horizon credit assignment:

| Hyperparameter | Baseline PPO | Tuned PPO | Rationale |
| :--- | :--- | :--- | :--- |
| **Total Timesteps** | 500,000 | 500,000 | Identical sample budget for fair comparison |
| **Entropy Coeff (`ent_coef`)** | `0.0` | **`0.01`** | Prevents premature entropy collapse; preserves exploration |
| **Optimization Epochs (`n_epochs`)** | `10` | **`4`** | Reduces gradient updates from 640 to 256/iter; avoids destructive policy drift |
| **Discount Factor (`gamma`)** | `0.99` | **`0.999`** | Lengthens planning horizon so the $+100$ terminal landing reward propagates back |
| **GAE Lambda (`gae_lambda`)** | `0.95` | **`0.98`** | Smooths advantage estimates over multi-hundred-step descent trajectories |
| **Learning Rate** | `3e-4` | `3e-4` | Standard Adam optimizer learning rate |
| **Rollout Steps (`n_steps`)** | `1024` | `1024` | 4 parallel environments $\times$ 1024 = 4096 samples per update |
| **Batch Size** | `64` | `64` | Mini-batch sample size per gradient step |
| **Policy Network** | `MlpPolicy` [64, 64] | `MlpPolicy` [64, 64] | Standard 2-layer MLP for actor and critic |

---

## 📊 Experimental Results

All evaluations were executed deterministically across fixed seed ranges for rigorous reproducibility.

### 1. Comparative Evaluation (Seeds 1000–1049, 50 Episodes)

Evaluated side-by-side using `evaluation/compare_training.py`:

| Agent Model | Mean Reward | Std Deviation | Episodes $\ge 200$ | Success Rate | Solved Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline PPO (500k)** | 98.97 | 118.80 | 14 / 50 | 28.0% | ❌ Unsolved |
| **Tuned PPO (500k)** | **219.05** | **20.12** | **41 / 50** | **82.0%** | **✅ SOLVED** |

* **Mean Reward Gain:** $+120.08$ points
* **Variance Reduction:** Standard deviation dropped from $118.80$ to $20.12$ (a $5.9\times$ reduction in volatility, eliminating erratic crash sequences).
* **Success Rate:** Jumped from $28.0\%$ to $82.0\%$ (almost $3\times$ higher).

---

### 2. Generalization Test on Unseen Seeds (Seeds 2000–2049, 50 Episodes)

To verify that the tuned agent did not overfit to the evaluation seeds, it was tested on 50 completely unseen seeds via `evaluation/validate_tuned.py`:

| Metric | Tuned PPO on Unseen Seeds | Target |
| :--- | :--- | :--- |
| **Mean Reward** | **213.18** | $\ge 200.0$ |
| **Std Deviation** | **28.99** | Low variance |
| **Successful Episodes ($\ge 200$)** | **37 / 50** | High consistency |
| **Success Rate** | **74.0%** | Robust generalization |

The agent maintained a mean return above the 200 threshold with strong landing stability, demonstrating robust generalization to unfamiliar terrain and wind perturbations.

---

## 📁 Repository Structure

```
├── environment/
│   └── custom-lander.py         # Custom environment extensions & wrappers
├── evaluation/
│   ├── compare_training.py      # Side-by-side comparison of baseline vs tuned models
│   ├── compare.py               # Checkpoint comparison utility
│   ├── evaluation.py            # Baseline evaluation script with visual rendering
│   └── validate_tuned.py        # Independent validation on unseen seed sets
├── models/
│   ├── baseline/                # Baseline checkpoint directory (best model)
│   ├── baseline_final.zip       # Final baseline model checkpoint (500k steps)
│   ├── tuned/                   # Tuned checkpoint directory (best model)
│   └── tuned_final.zip          # Final tuned model checkpoint (500k steps)
├── logs/
│   ├── baseline/                # Baseline evaluation logs (.npz)
│   ├── tuned/                   # Tuned evaluation logs (.npz)
│   └── tensorboard/             # TensorBoard training curves
├── training/
│   ├── baseline.py              # Baseline PPO training pipeline
│   ├── tuned.py                 # Tuned PPO training pipeline
│   ├── longer_training.py       # 1M step baseline experiment script
│   └── shape.py                 # Reward shaping experimentation script
├── .gitignore                   # Git exclusion rules
├── requirements.txt             # Pinned project dependencies
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and set up a Python virtual environment (Python 3.10–3.12 recommended):

```bash
git clone https://github.com/Amitkumar-21/lunar_RL.git
cd lunar_RL

# Create and activate virtual environment
python -m venv lunar
# Windows
.\lunar\Scripts\activate
# Linux/macOS
source lunar/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Evaluate Pretrained Models

To compare the baseline and tuned models on identical seeds:
```bash
python evaluation/compare_training.py
```

To validate the tuned agent on novel, unseen seeds:
```bash
python evaluation/validate_tuned.py
```

To run evaluation with real-time Pygame visualization:
```bash
python evaluation/evaluation.py
```

### 3. Train from Scratch

To train the baseline agent:
```bash
python training/baseline.py
```

To train the tuned agent:
```bash
python training/tuned.py
```

Monitor training runs live with TensorBoard:
```bash
tensorboard --logdir=./logs/tensorboard/
```

---

## 🔮 Limitations & Future Work

1. **Reward Shaping:** Future experiments will incorporate custom Gymnasium reward wrappers (`training/shape.py`) to explicitly incentivize fuel conservation, upright touchdown angle, and softer vertical velocities near the surface.
2. **Observation Normalization:** Integrating `VecNormalize` to standardize non-uniform observation scales (e.g. angle vs. position).
3. **Continuous Action Space:** Exploring `LunarLanderContinuous-v3` using PPO and SAC for continuous engine throttle control.
