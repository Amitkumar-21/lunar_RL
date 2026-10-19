
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym

import os
import zipfile

# Compatibility patch: Fix Python 3.12.0 zipfile.ZipExtFile.seek state bug for uncompressed streams
_orig_seek = zipfile.ZipExtFile.seek

def _patched_seek(self, offset, whence=os.SEEK_SET):
    res = _orig_seek(self, offset, whence)
    if getattr(self, "_compress_type", None) == zipfile.ZIP_STORED:
        self._compress_left = self._left
        if self._left > 0:
            self._eof = False
    return res

zipfile.ZipExtFile.seek = _patched_seek
ENV_ID = "LunarLander-v3"
SEEDS = range(1000, 1050)
EPISODES = len(SEEDS)

MODELS = {
    "Baseline (500k)": "./models/baseline_final.zip",
    "Tuned PPO (500k)": "./models/tuned_final.zip",
}   

def evaluate_model(name, path):
    model = PPO.load(path)
    rewards = []

    for seed in SEEDS:
        env = gym.make(ENV_ID)
        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward

        rewards.append(episode_reward)
        env.close()

    rewards = np.array(rewards)
    successes = np.sum(rewards >= 200)

    print(f"\n{name}")
    print("-" * 35)
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Std deviation: {rewards.std():.2f}")
    print(f"Successful episodes: {successes}/{EPISODES}")
    print(f"Success rate: {successes / EPISODES * 100:.1f}%")

    return rewards

if __name__ == "__main__":
    results = {}

    for name, path in MODELS.items():
        print(f"\nEvaluating {name}...")
        results[name] = evaluate_model(name, path)

    difference = results["Tuned PPO (500k)"].mean() - results["Baseline (500k)"].mean()

    print("\n========== COMPARISON ==========")
    print(f"Mean reward difference: {difference:+.2f}")
