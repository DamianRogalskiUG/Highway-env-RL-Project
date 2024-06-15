import gymnasium as gym
from stable_baselines3 import PPO
import highway_env

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
        "vehicles_count": 20,
        "duration": 120,
        "initial_spacing": 2,
        "collision_reward": -1,
        "reward_speed_range": [20, 30],
        "simulation_frequency": 60,
        "policy_frequency": 1,
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        "screen_width": 600,
        "screen_height": 150,
        "centering_position": [0.3, 0.5],
        "scaling": 5.5,
        "show_trajectories": False,
        "render_agent": True,
        "offscreen_rendering": False
    }
    env.configure(simulation_params)
    return env

def train_ppo_model(env, total_timesteps=int(2e4), num_cpu=8, batch_size=32):
    num_steps = batch_size * 12 // num_cpu
    policy_architecture = [dict(pi=[256, 256], vf=[256, 256])]
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=policy_architecture),
        n_steps=num_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        verbose=2,
        tensorboard_log="highway_ppo/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("highway_ppo/model")
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

def main():
    env = create_env()

    if TRAIN_MODEL:
        model = train_ppo_model(env)
    else:
        model = PPO.load("../PPO/highway_ppo/model.zip")

    test_ppo_model(model, env)

if __name__ == "__main__":
    main()
