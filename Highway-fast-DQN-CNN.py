import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import highway_env  # noqa: F401


def create_train_env():
    env = gym.make("highway-fast-v0")
    env.configure(
        {
            "observation": {
                "type": "GrayscaleObservation",
                "observation_shape": (128, 64),
                "stack_size": 4,
                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                "scaling": 1.75,
            },
        }
    )
    env.reset()
    return env


def create_test_env():
    env = create_train_env()
    env.configure({"policy_frequency": 15, "duration": 20})
    env.reset()
    return env


def train_dqn_model(env, total_timesteps=int(2e4)):
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
        tensorboard_log="highway_cnn/",
    )
    model.learn(total_timesteps=total_timesteps)
    model.save("highway_cnn/model")
    return model


def record_video(model, env, video_length):
    env = VecVideoRecorder(
        env,
        "highway_cnn/videos/",
        record_video_trigger=lambda x: x == 0,
        video_length=video_length,
        name_prefix="dqn-agent",
    )
    obs, info = env.reset()
    for _ in range(video_length + 1):
        action, _ = model.predict(obs)
        obs, _, _, _, _ = env.step(action)
    env.close()


def main():
    # Train
    train_env = create_train_env()
    model = train_dqn_model(train_env)

    # Record video
    model = DQN.load("highway_cnn/model")
    test_env = create_test_env()
    video_length = 2 * test_env.envs[0].config["duration"]
    record_video(model, test_env, video_length)


if __name__ == "__main__":
    main()
