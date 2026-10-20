
import gymnasium as gym

class CustomLunarLander(gym.Wrapper):
    """
    LunarLander with additional reward shaping.

    Preserves the original LunarLander reward and adds:
    - A small penalty for excessive speed near the ground.
    - A bonus for a stable, upright landing.
    """

    def __init__(self, render_mode=None):
        env = gym.make("LunarLander-v3", render_mode=render_mode)
        super().__init__(env)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Observation indices:
        # 0: horizontal position, 1: vertical position
        # 2: horizontal velocity, 3: vertical velocity
        # 4: angle, 5: angular velocity
        # 6: left leg contact, 7: right leg contact

        x_velocity = obs[2]
        y_velocity = obs[3]
        angle = obs[4]
        left_leg_contact = obs[6] > 0
        right_leg_contact = obs[7] > 0

        # Penalize excessive speed close to the ground.
        # Vertical position is higher when the lander is higher.
        if obs[1] < 0.5:
            reward -= 0.1 * (
                abs(x_velocity) + abs(y_velocity)
            )

        # Small bonus when the episode ends in a stable landing.
        successful_landing = (
            terminated
            and left_leg_contact
            and right_leg_contact
            and abs(angle) < 0.1
            and abs(x_velocity) < 0.2
            and abs(y_velocity) < 0.2
        )

        if successful_landing:
            reward += 10.0

        return obs, reward, terminated, truncated, info
