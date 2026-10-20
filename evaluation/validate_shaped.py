
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

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO


model = PPO.load("./models/shaped_final")
env = gym.make("LunarLander-v3")

rewards = []
successes = 0

for seed in range(2000, 2050):
    obs, info = env.reset(seed=seed)
    episode_reward = 0
    terminated = False
    truncated = False

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

    rewards.append(episode_reward)

    if episode_reward >= 200:
        successes += 1

env.close()

print("\n--- Shaped PPO: Unseen Seed Evaluation ---")
print(f"Mean reward: {np.mean(rewards):.2f}")
print(f"Reward standard deviation: {np.std(rewards):.2f}")
print(f"Successful episodes: {successes}/50")
print(f"Success rate: {successes / 50 * 100:.1f}%")
