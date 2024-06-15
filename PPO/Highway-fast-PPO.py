import gymnasium as gym
from stable_baselines3 import PPO
import highway_env

# Set to True if training the model is needed
TRAIN_MODEL = True  # Change to True if you want to train

def create_env():
    env = gym.make("highway-fast-v0", render_mode="rgb_array")
    simulation_params = {
        "simulation_frequency": 60,
        "lanes_count": 3,
        "vehicles_count": 30
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
        model = PPO.load("highway_ppo/model")

    test_ppo_model(model, env)

if __name__ == "__main__":
    main()
