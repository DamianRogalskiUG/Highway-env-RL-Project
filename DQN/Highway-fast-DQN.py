import gymnasium as gym
import numpy as np
from stable_baselines3 import DQN
import highway_env

TRAIN_MODEL = False  # Change to True if you want to train



def create_env():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    simulation_params = {
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 3,
        "vehicles_count": 30,
        "duration": 120,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 60,
        "policy_frequency": 2,
        "screen_width": 600,
        "screen_height": 150,
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }
    env.configure(simulation_params)
    return env


def train_dqn_model(env, total_timesteps=4e4):
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/"
    )
    model.learn(total_timesteps=int(total_timesteps))
    model.save("highway_dqn/model")

    return model


def test_trained_model(model, env, episodes=1000):
    for _ in range(episodes):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()

def main():
    env = create_env()


    if TRAIN_MODEL:
        model = train_dqn_model(env)
    else:
        model = DQN.load("highway_dqn/model", env=env)


    test_trained_model(model, env)

if __name__ == "__main__":
    main()
