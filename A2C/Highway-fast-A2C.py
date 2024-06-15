import gymnasium as gym
from stable_baselines3 import A2C
import highway_env

# Set to True if training the model is needed
TRAIN_MODEL = True  # Change to True if you want to train

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

def train_a2c_model(env, total_timesteps=int(2e4), num_cpu=8):
    policy_architecture = [256, 256]
    model = A2C(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=policy_architecture),
        n_steps=5,  # Number of steps to unroll for each update
        learning_rate=7e-4,
        verbose=1,  # Set to 1 for verbose output
        tensorboard_log="highway_a2c/"
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("highway_a2c/model")
    return model

def test_a2c_model(model, env, num_steps=500):
    obs, info = env.reset()
    for _ in range(num_steps):
        done = truncated = False
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()

def main():
    env = create_env()

    if TRAIN_MODEL:
        model = train_a2c_model(env)
    else:
        model = A2C.load("highway_a2c/model")

    test_a2c_model(model, env)

if __name__ == "__main__":
    main()
