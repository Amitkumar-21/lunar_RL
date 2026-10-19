
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

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

MODEL_PATH = "./models/baseline_final"

# Evaluate without rendering first
env = gym.make("LunarLander-v3")
model = PPO.load(MODEL_PATH)

rewards = []
successes = 0

for episode in range(50):
    obs, info = env.reset(seed=1000 + episode)
    total_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    rewards.append(total_reward)

    # Conventional LunarLander benchmark threshold
    if total_reward >= 200:
        successes += 1

    print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

print("\n--- Evaluation Results ---")
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Reward standard deviation: {np.std(rewards):.2f}")
print(f"Reward >= 200: {successes}/50")
print(f"Threshold rate: {successes / 50 * 100:.1f}%")

env.close()

# Watch one episode
env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset(seed=42)
total_reward = 0
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

print(f"Demo episode reward: {total_reward:.2f}")
env.close()
