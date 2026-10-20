
from environment.custom_lander import CustomLunarLander

env = CustomLunarLander()

obs, info = env.reset(seed=42)

print("Environment reset successfully")
print("Observation shape:", obs.shape)
print("Initial observation:", obs)

for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("Episode ended at step:", step + 1)
        break

env.close()
print("Custom environment test completed")
