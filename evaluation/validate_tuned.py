
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
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

MODEL_PATH = "./models/tuned_final"
ENV_ID = "LunarLander-v3"
SEEDS = range(2000, 2050)


def evaluate_model():
    model = PPO.load(MODEL_PATH)
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

    print("\nTuned PPO — Unseen Seed Validation")
    print("-" * 40)
    print(f"Episodes: {len(rewards)}")
    print(f"Mean reward: {rewards.mean():.2f}")
    print(f"Std deviation: {rewards.std():.2f}")
    print(f"Successful episodes: {successes}/{len(rewards)}")
    print(f"Success rate: {successes / len(rewards) * 100:.1f}%")


if __name__ == "__main__":
    evaluate_model()
