import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

import highway_env

if __name__ == "__main__":
    # Set to True to train the agent, False to load and run the trained agent
    train_agent = False

    if train_agent:
        # Training hyperparameters
        num_cpu = 8
        batch_size = 64
        total_timesteps = int(2e4)
        num_steps = batch_size * 12 // num_cpu

        # Create a vectorized environment
        env_name = "highway-fast-v0"
        env = make_vec_env(env_name, n_envs=num_cpu, vec_env_cls=SubprocVecEnv)

        # Define the policy network architecture and initialize the PPO agent
        policy_architecture = [dict(pi=[256, 256], vf=[256, 256])]
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=policy_architecture),
            n_steps=num_steps,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log="highway_ppo/",
        )

        # Train the agent
        model.learn(total_timesteps=total_timesteps)

        # Save the trained agent
        model.save("highway_ppo/model")

    # Load the trained agent
    model = PPO.load("highway_ppo/model")

    # Create the environment for testing
    env_name = "highway-fast-v0"
    env = gym.make(env_name, render_mode="rgb_array")

    # Configure the environment
    simulation_params = {
        "simulation_frequency": 60,
        "lanes_count": 4,
        "vehicles_count": 30
    }
    env.configure(simulation_params)

    # Reset the environment and get initial observations
    obs, info = env.reset()

    # Run the simulation for a fixed number of steps
    num_steps = 500
    for _ in range(num_steps):
        done = truncated = False

        # Loop until episode ends or is truncated
        while not (done or truncated):
            # Agent predicts action based on observation
            action, _ = model.predict(obs)

            # Take action and observe the next state and reward
            obs, reward, done, truncated, info = env.step(action)

            # Render the environment
            env.render()
