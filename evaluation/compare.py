# larger evaluation on unseen shaping ppo model


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
import csv
import numpy as np
from stable_baselines3 import PPO




MODELS = {
    "Tuned PPO": "./models/tuned_final",
    "Reward-Shaped PPO": "./models/shaped_final",
}

SEEDS = range(3000, 3500)
SUCCESS_THRESHOLD = 200


def evaluate_model(model_path, env, seeds):
    model = PPO.load(model_path)
    results = []

    for seed in seeds:
        obs, info = env.reset(seed=seed)
        episode_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        results.append({
            "seed": seed,
            "reward": episode_reward,
            "success": int(episode_reward >= SUCCESS_THRESHOLD),
        })

    return results


def main():
    env = gym.make("LunarLander-v3")
    all_results = {}

    try:
        for name, model_path in MODELS.items():
            print(f"\nEvaluating {name}...")

            results = evaluate_model(model_path, env, SEEDS)
            all_results[name] = results

            rewards = [result["reward"] for result in results]
            successes = sum(result["success"] for result in results)

            print(f"Mean reward: {np.mean(rewards):.2f}")
            print(f"Std deviation: {np.std(rewards):.2f}")
            print(f"Successful episodes: {successes}/{len(results)}")
            print(f"Success rate: {successes / len(results) * 100:.1f}%")

        tuned = all_results["Tuned PPO"]
        shaped = all_results["Reward-Shaped PPO"]

        differences = [
            shaped[i]["reward"] - tuned[i]["reward"]
            for i in range(len(SEEDS))
        ]

        print("\n--- Paired Comparison ---")
        print(f"Mean reward difference (shaped - tuned): {np.mean(differences):.2f}")
        print(
            "Seeds where shaped PPO scored higher: "
            f"{sum(d > 0 for d in differences)}/{len(differences)}"
        )

        os.makedirs("results", exist_ok=True)

        with open(
            "results/large_evaluation.csv",
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["seed", "model", "reward", "success"])

            for name, results in all_results.items():
                for result in results:
                    writer.writerow([
                        result["seed"],
                        name,
                        round(result["reward"], 4),
                        result["success"],
                    ])

        print("\nDetailed results saved to results/large_evaluation.csv")

    finally:
        env.close()


if __name__ == "__main__":
    main()
