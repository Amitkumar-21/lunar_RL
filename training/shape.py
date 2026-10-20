from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from environment.custom_lander import CustomLunarLander


def make_custom_env():
    return CustomLunarLander()

# Training environment: 4 parallel environments
train_env = make_vec_env(make_custom_env,n_envs=4,seed=42,)

# Evaluation environment
eval_env = make_custom_env()

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./models/shaped/",
    log_path="./logs/shaped/",
    eval_freq=10_000,
    n_eval_episodes=10,
    deterministic=True,
)

model = PPO(
    "MlpPolicy",
    train_env,
    learning_rate=3e-4,
    n_steps=1024,
    batch_size=64,
    gamma=0.999,
    gae_lambda=0.98,
    ent_coef=0.01,
    n_epochs=4,
    seed=42,
    verbose=1,
    tensorboard_log="./logs/tensorboard/",
)

model.learn(
    total_timesteps=500_000,
    callback=eval_callback,
)

model.save("./models/shaped_final")

train_env.close()
eval_env.close()

print("Reward-shaped training completed.")
