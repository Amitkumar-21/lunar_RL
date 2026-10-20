
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from environment.custom_lander import CustomLunarLander
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

NUM_EPISODES = 100
SEED_START = 4000


def main():
    model = PPO.load("./models/shaped_final")
    env = CustomLunarLander()

    penalty_steps = 0
    bonus_episodes = 0
    total_episodes = 0
    episode_rewards = []

    for seed in range(SEED_START, SEED_START + NUM_EPISODES):
        obs, info = env.reset(seed=seed)
        terminated = False
        truncated = False
        episode_reward = 0.0
        episode_penalty_steps = 0
        episode_bonus_received = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)

            # Reproduce the reward conditions in custom_lander.py.
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if obs[1] < 0.5:
                episode_penalty_steps += 1

            # This diagnostic independently checks the current bonus condition.
            if (
                terminated
                and obs[6] > 0
                and obs[7] > 0
                and abs(obs[4]) < 0.1
                and abs(obs[2]) < 0.2
                and abs(obs[3]) < 0.2
            ):
                episode_bonus_received = True

        penalty_steps += episode_penalty_steps
        bonus_episodes += int(episode_bonus_received)
        total_episodes += 1
        episode_rewards.append(episode_reward)

    env.close()

    print("\n--- Reward Function Diagnostic ---")
    print(f"Episodes: {total_episodes}")
    print(f"Total steps penalized by height condition: {penalty_steps}")
    print(f"Episodes receiving stable-landing bonus: {bonus_episodes}")
    print(f"Mean episode reward: {np.mean(episode_rewards):.2f}")
    print(f"Successful episodes (reward >= 200): "
          f"{sum(r >= 200 for r in episode_rewards)}/{total_episodes}")


if __name__ == "__main__":
    main()
