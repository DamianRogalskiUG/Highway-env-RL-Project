import gymnasium as gym
from stable_baselines3 import DQN
import highway_env
import matplotlib.pyplot as plt
import numpy as np

# Set to True if training the model is needed
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
        "vehicles_count": 60,
        "duration": 120,
        "simulation_frequency": 30,
        "screen_width": 600,
        "screen_height": 150,
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }
    env.configure(simulation_params)
    return env


def train_dqn_model(env, total_timesteps=2e4):
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
        tensorboard_log="DQN/highway_dqn/"
    )

    # List to keep track of rewards
    rewards = []

    # Callback function to collect rewards
    def reward_callback(locals, globals):
        rewards.append(locals['rewards'])
        return True

    model.learn(total_timesteps=int(total_timesteps), callback=reward_callback)
    model.save("DQN/highway_dqn/model")

    # Save rewards for plotting
    with open("DQN/highway_dqn/rewards.npy", "wb") as f:
        np.save(f, rewards)

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


def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards over Time')
    plt.savefig('DQN.jpg')
    plt.show()


def main():
    env = create_env()

    if TRAIN_MODEL:
        model = train_dqn_model(env)

        # Load rewards
        with open("highway_dqn/rewards.npy", "rb") as f:
            rewards = np.load(f)

        plot_rewards(rewards)
    else:
        model = DQN.load("DQN/highway_dqn/model", env=env)

    test_trained_model(model, env)


if __name__ == "__main__":
    main()
