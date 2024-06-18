import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import highway_env
import matplotlib.pyplot as plt
import numpy as np

# Set to True if training the model is needed
TRAIN_MODEL = False  # Change to True if you want to train


def create_env():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    env.configure(
        {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,

            },
            "simulation_frequency": 30,
            "vehicles_count": 15,

        }
    )
    env.reset()
    return env


def create_test_env():
    env = create_env()
    env.configure({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env


def train_dqn_model(env, total_timesteps=int(1e5)):
    model = DQN(
        "CnnPolicy",
        DummyVecEnv([lambda: env]),
        learning_rate=5e-4,
        buffer_size=15000,
        learning_starts=200,
        batch_size=32,
        gamma=0.8,
        train_freq=1,
        gradient_steps=1,
        target_update_interval=50,
        exploration_fraction=0.7,
        verbose=1,
        tensorboard_log="DQN+CNN/highway_cnn/",
    )

    # List to keep track of rewards
    rewards = []

    # Callback function to collect rewards
    def reward_callback(locals, globals):
        rewards.append(locals['rewards'][0])
        return True

    model.learn(total_timesteps=total_timesteps, callback=reward_callback)
    model.save("DQN+CNN/highway_cnn/model")

    # Save rewards for plotting
    with open("DQN+CNN/highway_cnn/rewards.npy", "wb") as f:
        np.save(f, rewards)

    return model


def record_video(model, env, video_length):
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()


def test_cnn_model(model, env, num_steps=500):
    obs, info = env.reset()
    for _ in range(num_steps):
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()


def plot_rewards(rewards):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards)
    plt.xlabel('Episodes')
    plt.ylabel('Rewards')
    plt.title('Rewards over Time')
    plt.savefig('DQN+CNN.jpg')
    plt.show()


def main():
    env = create_env()
    if TRAIN_MODEL:
        model = train_dqn_model(env)

        # Load rewards
        with open("DQN+CNN/highway_cnn/rewards.npy", "rb") as f:
            rewards = np.load(f)

        plot_rewards(rewards)
    else:
        model = DQN.load("DQN+CNN/highway_cnn/model")
        test_cnn_model(model, env)


if __name__ == "__main__":
    main()
