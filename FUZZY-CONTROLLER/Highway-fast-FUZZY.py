import gymnasium as gym
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import highway_env


def create_env():
    env = gym.make("highway-v0", render_mode="rgb_array")
    simulation_params = {
        "observation": {
            "type": "Kinematics"
        },
        "action": {
            "type": "DiscreteMetaAction",
        },
        "lanes_count": 4,
        "vehicles_count": 10,
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


def define_fuzzy_controller():
    # Define fuzzy variables
    distance = ctrl.Antecedent(np.arange(0, 101, 1), 'distance')
    speed = ctrl.Antecedent(np.arange(0, 101, 1), 'speed')
    action = ctrl.Consequent(np.arange(0, 5, 1), 'action')

    # Define membership functions for distance
    distance['very_close'] = fuzz.trapmf(distance.universe, [0, 0, 10, 20])
    distance['close'] = fuzz.trapmf(distance.universe, [10, 20, 30, 40])
    distance['medium'] = fuzz.trapmf(distance.universe, [30, 40, 50, 60])
    distance['far'] = fuzz.trapmf(distance.universe, [50, 60, 70, 80])
    distance['very_far'] = fuzz.trapmf(distance.universe, [70, 80, 100, 100])

    # Define membership functions for speed
    speed['very_slow'] = fuzz.trapmf(speed.universe, [0, 0, 10, 20])
    speed['slow'] = fuzz.trapmf(speed.universe, [10, 20, 30, 40])
    speed['average'] = fuzz.trapmf(speed.universe, [30, 40, 50, 60])
    speed['fast'] = fuzz.trapmf(speed.universe, [50, 60, 70, 90])
    speed['very_fast'] = fuzz.trapmf(speed.universe, [90, 95, 100, 100])

    # Define membership functions for action
    action['brake_hard'] = fuzz.trimf(action.universe, [0, 0, 1])
    action['brake'] = fuzz.trimf(action.universe, [0, 1, 2])
    action['keep_lane'] = fuzz.trimf(action.universe, [1, 2, 3])
    action['change_left'] = fuzz.trimf(action.universe, [2, 3, 4])
    action['change_right'] = fuzz.trimf(action.universe, [3, 4, 4])

    # Define fuzzy rules for lane change decisions
    rules = [
        ctrl.Rule(distance['very_close'] & speed['very_fast'], action['brake_hard']),
        ctrl.Rule(distance['very_close'] & speed['fast'], action['brake_hard']),
        ctrl.Rule(distance['very_close'] & speed['average'], action['brake']),
        ctrl.Rule(distance['very_close'] & speed['slow'], action['brake']),
        ctrl.Rule(distance['very_close'] & speed['very_slow'], action['brake']),

        ctrl.Rule(distance['close'] & speed['very_fast'], action['brake_hard']),
        ctrl.Rule(distance['close'] & speed['fast'], action['brake']),
        ctrl.Rule(distance['close'] & speed['average'], action['brake']),
        ctrl.Rule(distance['close'] & speed['slow'], action['keep_lane']),
        ctrl.Rule(distance['close'] & speed['very_slow'], action['keep_lane']),

        ctrl.Rule(distance['medium'] & speed['very_fast'], action['brake']),
        ctrl.Rule(distance['medium'] & speed['fast'], action['keep_lane']),
        ctrl.Rule(distance['medium'] & speed['average'], action['keep_lane']),
        ctrl.Rule(distance['medium'] & speed['slow'], action['change_left']),
        ctrl.Rule(distance['medium'] & speed['very_slow'], action['change_left']),

        ctrl.Rule(distance['far'] & speed['very_fast'], action['keep_lane']),
        ctrl.Rule(distance['far'] & speed['fast'], action['keep_lane']),
        ctrl.Rule(distance['far'] & speed['average'], action['change_right']),
        ctrl.Rule(distance['far'] & speed['slow'], action['change_right']),
        ctrl.Rule(distance['far'] & speed['very_slow'], action['change_right']),

        ctrl.Rule(distance['very_far'] & speed['very_fast'], action['keep_lane']),
        ctrl.Rule(distance['very_far'] & speed['fast'], action['keep_lane']),
        ctrl.Rule(distance['very_far'] & speed['average'], action['change_right']),
        ctrl.Rule(distance['very_far'] & speed['slow'], action['change_right']),
        ctrl.Rule(distance['very_far'] & speed['very_slow'], action['change_right']),
    ]

    # Create the control system
    action_ctrl = ctrl.ControlSystem(rules)
    action_sim = ctrl.ControlSystemSimulation(action_ctrl)

    return action_sim


def get_action(obs, action_sim):

    ego_vehicle = obs[0]  # First row as ego vehicle
    lead_vehicle = obs[1]  # Second row as lead vehicle

    distance_to_lead = lead_vehicle[2] * 100  # Scale to 0-100
    ego_speed = ego_vehicle[3] * 100  # Scale to 0-100

    action_sim.input['distance'] = distance_to_lead
    action_sim.input['speed'] = ego_speed

    action_sim.compute()

    return int(action_sim.output['action'])


def test_fuzzy_controller(env, action_sim, episodes=10):
    for episode in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        while not (done or truncated):
            action = get_action(obs, action_sim)
            obs, reward, done, truncated, info = env.step(action)
            env.render()
    env.close()


def main():
    env = create_env()
    controller_sim = define_fuzzy_controller()
    test_fuzzy_controller(env, controller_sim)


if __name__ == "__main__":
    main()
