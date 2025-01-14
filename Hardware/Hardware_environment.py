import gymnasium
import numpy as np
import requests
import time
from stable_baselines3 import PPO
from gymnasium import spaces
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import random
import sys
sys.path.append('./Hardware')

import Hardware_reward

import warnings
warnings.filterwarnings("ignore")
from math import sin, cos

# Constants
ANGLE_MIN, ANGLE_MAX = -0.122173, 0.122173
beam_length = 2
setpoint = 0
dt = 0.2
g = 9.82
r = 0.01975                      
kg = 0.035
I = 2/5*(kg)*(r**2)


rewards = []
actions = []
positions = []
set_angles = []
ball_speeds = []
step_numbers = []  

def send_set_angle(set_angle, post_url):
    angle_deg = np.degrees(set_angle)
    payload1 = {"angle_set": int(angle_deg)}
    # print(f'Set angle {angle_deg} URL= {post_url}')
    response = requests.post(post_url, json=payload1)
    return None

def get_data(get_url):
    try:
        # print(f'Get url : {get_url}')
        response = requests.get(get_url)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error getting data: {e}")
        return None
    
def plot():
    print('Ploting........')
    fig, axs = plt.subplots(5, 1, figsize=(22, 7))  
    x_length = range(len(set_angles))
    # Plot reward
    axs[0].plot(x_length, rewards, label='Reward')
    axs[0].set_title('Reward over Time')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('Reward')
    #axs[0].set_ylim(-1, 1)  
 
    # Plot ball position
    axs[1].plot(x_length, positions, label='Position')
    axs[1].set_title('Ball Position')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Position')
    #axs[1].set_ylim(-1, 1) 

    #Plot speed
    axs[2].plot(x_length, ball_speeds, label='Speed')
    axs[2].set_title('Ball speed')
    axs[2].set_xlabel('Step')
    axs[2].set_ylabel('speed')
    #axs[2].set_ylim(-1, 1) 

    #Plot speed
    axs[3].plot(x_length, set_angles, label='angles')
    axs[3].set_title('Set angles')
    axs[3].set_xlabel('Step')
    axs[3].set_ylabel('angle')
    #axs[3].set_ylim(-1, 1) 

    #Plot speed
    axs[4].plot(x_length, actions, label='Actions')
    axs[4].set_title('Actions')
    axs[4].set_xlabel('Step')
    axs[4].set_ylabel('Actions')
    #axs[4].set_ylim(-1, 1) 

    plt.tight_layout()
    plt.show()  

class HardwareEnv(gymnasium.Env):
    
    step_number = 0

    def __init__(self, get_url, post_url, max_steps=1000, tolerance=0.01, stability_steps=20, testing_agent=False):
        super(HardwareEnv, self).__init__()
        
        self.get_url = get_url
        self.post_url = post_url
        self.testing_agent = testing_agent
        
        self.action_space = spaces.Discrete(3)  
        self.observation_space = spaces.Box(
            low=np.array([ANGLE_MIN, -np.inf, -np.inf, -beam_length / 2]),
            high=np.array([ANGLE_MAX, np.inf, np.inf, beam_length / 2]),
            dtype=np.float32
        )

        self.max_steps = max_steps
        self.tolerance = tolerance
        self.stability_steps = stability_steps
        self.stable_count = 0
        self.step_count = 0
        self.angle = 0.0
        self.angle_change_speed = ANGLE_MAX*4
        self.timestep=0.05
        
        self.state = np.zeros(4)
        self.set_angle = 0  

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
        # self.state = np.zeros(4)
        self.state = np.zeros(4, dtype=np.float32)
        self.set_angle = 6 #random.choice([-ANGLE_MAX,-ANGLE_MAX/2, ANGLE_MAX/2, ANGLE_MAX])
        send_set_angle(self.set_angle, post_url=self.post_url)
        print(f'Resetting to angle: {self.set_angle}')
        time.sleep(1)
        self.step_count = 0
        self.stable_count = 0
        return self.state, {}
    
    # def calculate_reward(self, setpoint, position, beam_length, action):
    #     R1 = (1 - abs(setpoint - position))**2
    #     if position > 0:
    #         if action == 0:
    #             R2 = R1*1.2
    #         else:
    #             R2 = R1*0.8

    #     elif position < 0:
    #         if action == 2:
    #             R2 = R1*1.2
    #         else:
    #             R2 = R1*0.8

    #     else:
    #         R2 = R1*1.3

    #     return max(0, min(1, R2))
    
    def _action_conversion1(self, action, setpoint, position):
        # 0  --> -   
        # 1  --> remain same
        # 2  --> +
        diff = abs(setpoint - position)
        diff = diff * 0.9 * ANGLE_MAX
        angle1 = min(diff, ANGLE_MAX)
        action_angles = [-angle1, 0, +angle1]  

        final_action = action_angles[action]
        angle_deg = np.degrees(final_action)

        return action_angles[action]
    

    def step(self, action):
        self.step_number += 1
        step_numbers.append(self.step_number)
        #print(f'get_url before sending:   {self.get_url}')
        sensor_data = get_data(get_url=self.get_url)
        if sensor_data:
            ball_position = sensor_data['position']
            angle = sensor_data['angle_set']
            angle_current = sensor_data['angle_current']
            # ball_speed = sensor_data['speed']
            

            ball_speed = -g/(1 + I)*sin(angle)*dt
            ball_speed = +ball_speed          
            

            angle = np.radians(angle)
            state = np.array([angle, ball_position, ball_speed, setpoint], dtype=np.float32)
            self.state = state

            self.set_angle = self._action_conversion1(action, setpoint, ball_position)

            send_set_angle(self.set_angle, post_url=self.post_url)

            # reward = self.calculate_reward(setpoint, ball_position, beam_length, action)
            reward = Hardware_reward.calculate_reward(setpoint, ball_position, beam_length, action)

            self.step_count += 1
            
            done = False
            if not self.testing_agent:
                if abs(setpoint - ball_position) < self.tolerance and abs(ball_speed) < self.tolerance:
                    self.stable_count += 1
                    if self.stable_count >= self.stability_steps:
                        done = True  # Ball stable near center
                        reward += 1  # Bonus for achieving stability
                        print('Reset due to stability')
                        send_set_angle(self.set_angle, post_url=self.post_url)
                        print('Setting angle ')
                        time.sleep(1)

                else:
                    self.stable_count = 0

            if self.step_count >= self.max_steps:
                done = True  # Max steps reached
                print('Reset because max step reached')
                self.set_angle = ANGLE_MAX
                send_set_angle(self.set_angle)
                print(f'Setting angle {self.set_angle}')
                time.sleep(1)


            angle_p, ball_position_p, ball_speed_p, setpoint_p = state
            rewards.append(reward)
            positions.append(ball_position_p)  
            set_angles.append(angle_p)  
            ball_speeds.append(ball_speed_p)  
            actions.append(action)
            step_numbers.append(self.step_number)


            print(f'set_angle:  {np.degrees(self.set_angle):.2f}, angle_current: {angle_current},  reward:  {reward:.2f},  position:  {ball_position:.2f}, speed:  {ball_speed:.2f} ')
            time.sleep(0.05)
            return state, reward, done, False, {}

        else:
            print('No data received from hardware')
            return self.state, 0, False, {}
        
