import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import highway_env
import matplotlib.pyplot as plt
import numpy as np

# Set to True if training the model is needed
TRAIN_MODEL = True # Change to True if you want to train


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
        "vehicles_count": 40,
        "duration": 120,
        "simulation_frequency": 30,
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
    }
    env.configure(simulation_params)
    return env


def train_ppo_model(env, total_timesteps=int(1e5), num_cpu=6, batch_size=64):
    num_steps = batch_size * 12 // num_cpu
    policy_architecture = [dict(pi=[256, 256], vf=[256, 256])]
    new_env = make_vec_env("highway-fast-v0", n_envs=num_cpu, vec_env_cls=SubprocVecEnv)
    model = PPO(
        "MlpPolicy",
        new_env,
        policy_kwargs=dict(net_arch=policy_architecture),
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        verbose=2,
        tensorboard_log="../PPO/highway_ppo/"
    )

    # List to keep track of rewards
    rewards = []

    # Callback function to collect rewards
    def reward_callback(locals, globals):
        rewards.append(locals['rewards'])
        return True

    model.learn(total_timesteps=total_timesteps, callback=reward_callback)
    model.save("../PPO/highway_ppo/model")

    # Save rewards for plotting
    with open("../PPO/highway_ppo/rewards.npy", "wb") as f:
        np.save(f, rewards)
        plot_rewards(rewards)

    return model


def test_ppo_model(model, env, num_steps=500):
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
    plt.savefig('../PPO/PPO_20k_episodes.jpg')
    plt.show()


def main():
    env = create_env()

    if TRAIN_MODEL:
        model = train_ppo_model(env)

    else:
        model = PPO.load("../PPO/highway_ppo/model")

    test_ppo_model(model, env)


if __name__ == "__main__":
    main()
