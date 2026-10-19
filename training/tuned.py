
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback

import os

os.makedirs("./models/tuned/", exist_ok=True)
os.makedirs("./logs/tuned/", exist_ok=True)
os.makedirs("./logs/tensorboard/tuned/", exist_ok=True)

# Create training environment
env = make_vec_env("LunarLander-v3", n_envs=4, seed=42)

# Create evaluation environment
eval_env = make_vec_env("LunarLander-v3", n_envs=1, seed=100)

# Evaluate periodically and save the best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/tuned/",
    log_path="./logs/tuned/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

# Initialize PPO
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    n_epochs=4,
    verbose=1,
    tensorboard_log="./logs/tensorboard/tuned/",
    seed=42,
)

# Train the agent
model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
    progress_bar=True,
)

# Save final model
model.save("./models/tuned_final")

env.close()
eval_env.close()

print("Tuned training completed!")
