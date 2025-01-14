import tensorflow as tf
import gym
import os
import sys
sys.path.append('./Simulation')
import ballbeam_gym
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import matplotlib.pyplot as plt
base_dir = 'Models'

class RewardCallback(BaseCallback):
    def __init__(self, training_rewards_storage, render, env1, verbose=1):
        super(RewardCallback, self).__init__(verbose)

        self.training_rewards_storage = training_rewards_storage
        self.actions_angles_log = []
        self.render = render
        self.env1 = env1
    def _on_step(self) -> bool:
        # Collect rewards from the environment
        if self.render==True:
            #print(f'render called here \n')
            self.env1.render()
        for reward in self.locals['rewards']:
            self.training_rewards_storage.append(reward)

        action = self.locals['actions'][0]
        #print(' ')
        #print(f'here i am printing locals   : {self.locals} /n')
        angle = self.locals['new_obs'][0][0]
        self.actions_angles_log.append((action, angle))

        return True

    def print_action_angle_table(self):
        print("Action | Angle")
        print("--------------")
        print(self.actions_angles_log)
        for action, angle in self.actions_angles_log:
            print(f"{action}     | {angle}")


def train_simulation_rl_agent(num_steps, episode_length, render, Use_Reward_function_advance, Setpoint, Use_pre_trained_agent, load_path, save_path ):
    # Environment arguments
    kwargs = {
        'timestep': 0.05, 
        'setpoint': Setpoint,
        'beam_length': 2.0,
        'max_angle': 0.122173,
        'init_velocity': 0.0,
        'action_mode': 'discrete',
        'max_timesteps': 300,
        'Use_Reward_function_advance': Use_Reward_function_advance
    }
    if Setpoint is not None: 
        print(f'Setpoint is set to {Setpoint}...')
    else:
        print('Setpoint is None...')
    # Create environment
    env1 = gym.make('BallBeamSetpoint-v0', **kwargs)

    # Wrap the environment in a DummyVecEnv
    env = DummyVecEnv([lambda: env1])
    #env1.render()
    # Initialize the rewards storage
    training_rewards_storage = []

    # Define the model
    model = PPO('MlpPolicy', env, verbose=1)
    if Use_pre_trained_agent:
        model = PPO.load(load_path, env=env)
        print('Model loaded sccessfully ... ')

    # Initialize the callback with rewards storage
    reward_callback = RewardCallback(training_rewards_storage, render, env1, verbose=1)

    # Train the model
    model.learn(total_timesteps=num_steps, callback=reward_callback)


    from datetime import datetime
    import os

    # Set the model name as desired
    subfolder_path = save_path
    n = 1
    while os.path.exists(subfolder_path):
        subfolder_name = f"{subfolder_path}_{n}"
        subfolder_path = os.path.join(subfolder_name)
        n += 1

    os.makedirs(subfolder_path)

    save_path = os.path.join(subfolder_path, "model.zip")  
    model.save(save_path)

    # Plotting the rewards
    plt.plot(training_rewards_storage) #[-1000:])
    plt.xlabel('Steps')
    plt.ylabel('Rewards')
    plt.title('Training Rewards over Time')
    plt.show()

    env1.plot_logs(subfolder_path)
    #env1.scatter_plot_metrics()

def test_simulation_rl_agent(render, num_steps, Setpoint, load_path):
    # Environment arguments
    kwargs = {
        'timestep': 0.05, 
        'setpoint': Setpoint,
        'beam_length': 2.0,
        'max_angle': 0.122173,
        'init_velocity': 0.0,
        'action_mode': 'discrete',
        'max_timesteps': 300
    }

    env1 = gym.make('BallBeamSetpoint-v0', **kwargs)
    env = DummyVecEnv([lambda: env1])

    # Load the trained model
    model = PPO('MlpPolicy', env, verbose=1)
    path_to_load = os.path.join(load_path, "model.zip")
    model = PPO.load(path_to_load)

    # Reset the environment to get the initial observation
    obs = env.reset()
    testing_actions_angles_log = []

    # Test the trained agent for `num_steps`
    print('Testing the agent')
    for i in range(num_steps):
        # Predict the action
        action, _ = model.predict(obs)

        # Step through the environment
        obs, reward, done, info = env.step(action)

        # Log the results
        position = obs[0][1]
        angle = obs[0][2]
        testing_actions_angles_log.append((position, action, angle))

        # Render the environment if the `render` flag is set to True
        
        env1.render()

        # Reset the environment if the episode is done
        if done:
            obs = env.reset()
            print("Episode finished.")
            print("position | Action | Angle")
            print("--------------")
            for position, action, angle in testing_actions_angles_log:
                print(f"{position}   |   {action}     |   {angle}")

            testing_actions_angles_log = []
