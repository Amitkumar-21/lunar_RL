
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

def main():
    env = gym.make("LunarLander-v3", render_mode="human")
    model = PPO.load("models/shaped_final.zip")
    episodes = 5
    try:
        for episode in range(episodes):
            obs, info = env.reset(seed=4000 + episode)
            terminated = False
            truncated = False
            total_reward = 0.0
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
            print(
                f"Episode {episode + 1}: "
                f"Reward = {total_reward:.2f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
