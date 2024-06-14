import gymnasium as gym
import numpy as np
from gym.wrappers import RecordVideo
from matplotlib import pyplot as plt
from stable_baselines3 import DQN
import highway_env

# Set to True if training the model is needed
TRAIN = False  # Change to True if you want to train

def main():
    # Set up the environment
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.reset()

    # Create the model
    model = DQN(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[256, 256]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        verbose=1,
        tensorboard_log="highway_dqn/",

    )

    # Lists for storing training statistics
    episode_rewards = []
    mean_rewards = []

    # Train the model
    if TRAIN:
        # Callback function to save stats after each episode
        def _save_stats(locals_, globals_):
            nonlocal episode_rewards, mean_rewards
            episode_rewards.append(locals_['self'].episode_reward)
            mean_rewards.append(np.mean(episode_rewards[-100:]))

        model.learn(total_timesteps=int(2e4), callback=_save_stats)

    # Load the trained model
    model = DQN.load("highway_dqn/model", env=env)
    env.configure({
        "simulation_frequency": 60,
        "lanes_count": 3,
        "vehicles_count": 30
    })

    # Function to save episode rewards during training
    def _save_stats(locals_, globals_):
        nonlocal episode_rewards, mean_rewards
        episode_rewards.append(env.episode_reward)
        mean_rewards.append(np.mean(episode_rewards[-100:]))

    for _ in range(1000):
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, truncated, info = env.step(action)
            # Render
            env.render()

    env.close()

if __name__ == "__main__":
    main()
