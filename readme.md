# LunarLander-v3 — PPO & Reward Shaping

Optimizing Proximal Policy Optimization (PPO) on Farama Foundation's **LunarLander-v3** using Stable-Baselines3 and PyTorch. 

The environment is officially considered solved at a mean score of **$\ge 200$**. Through hyperparameter tuning and custom reward shaping, performance improved from **28% to a perfect 100% success rate on unseen seeds**.

---

## 📊 Results Summary

All models trained for 500k timesteps and evaluated on the official, unshaped `LunarLander-v3` environment across 50 deterministic seeds:

| Stage / Model | Evaluated Seeds | Mean Reward | Std Dev | Success Rate ($\ge 200$) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Baseline PPO** | Test (1000–1049) | 98.97 | 118.80 | 28.0% (14/50) | ❌ Unsolved |
| **2. Tuned PPO** | Test (1000–1049) | 219.05 | 20.12 | 82.0% (41/50) | ✅ Solved |
| **3. Tuned PPO** | Unseen (2000–2049) | 213.18 | 28.99 | 74.0% (37/50) | ✅ Robust |
| **4. Shaped PPO** | Test (1000–1049) | 240.84 | 25.22 | 98.0% (49/50) | 🏆 Master |
| **5. Shaped PPO** | **Unseen (2000–2049)** | **240.57** | **20.34** | **100.0% (50/50)** | 🏆 **Flawless** |

---

## ⚙️ Key Hyperparameters

| Parameter | Baseline | Tuned & Shaped | Why It Matters |
| :--- | :--- | :--- | :--- |
| **`ent_coef`** | `0.0` | **`0.01`** | Prevents premature entropy collapse; preserves exploration |
| **`n_epochs`** | `10` | **`4`** | Reduces gradient updates from 640 to 256/iter; stops policy churn |
| **`gamma`** | `0.99` | **`0.999`** | Lengthens credit assignment for the $+100$ touchdown bonus |
| **`gae_lambda`**| `0.95` | **`0.98`** | Smooths advantage estimates over long descent trajectories |

### Reward Shaping (`environment/custom_lander.py`):
- **Velocity Dampening:** Penalizes horizontal and vertical speed when altitude $y < 0.5$ (`-0.1 * (|vx| + |vy|)`).
- **Soft Touchdown Bonus:** $+10.0$ bonus when landing upright with $|v| < 0.2$ and both legs touching down.

---

## 🚀 Quickstart

### 1. Installation
```bash
git clone https://github.com/Amitkumar-21/lunar_RL.git
cd lunar_RL
python -m venv lunar
.\lunar\Scripts\activate   # Linux/macOS: source lunar/bin/activate
pip install -r requirements.txt
```

### 2. Evaluation
```bash
# Evaluate shaped PPO on 50 unseen seeds (100% success rate)
python -m evaluation.validate_shaped

# Compare Baseline vs Tuned vs Shaped
python -m evaluation.compare_training
```

### 3. Training
```bash
# Train baseline PPO
python training/baseline.py

# Train tuned PPO
python training/tuned.py

# Train reward-shaped PPO
python training/shape.py
```

---

## 📁 Project Structure

```
├── environment/
│   └── custom_lander.py         # Gymnasium reward shaping wrapper
├── evaluation/
│   ├── compare_training.py      # Multi-model comparative benchmark
│   ├── evaluate_shaped.py       # Shaped model evaluation (test seeds)
│   ├── validate_shaped.py       # Shaped model validation (unseen seeds: 100%)
│   └── validate_tuned.py        # Tuned model validation (unseen seeds)
├── models/
│   ├── baseline_final.zip       # Baseline checkpoint (500k)
│   ├── tuned_final.zip          # Tuned checkpoint (500k)
│   └── shaped_final.zip         # Shaped checkpoint (500k)
├── training/
│   ├── baseline.py              # Baseline training pipeline
│   ├── tuned.py                 # Tuned training pipeline
│   └── shape.py                 # Reward-shaped training pipeline
├── requirements.txt
└── README.md
```
