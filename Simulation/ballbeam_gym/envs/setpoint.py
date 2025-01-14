""" 
Setpoint Environments

Environments where the objective is to keep the ball close to a set beam postion

BallBeamSetpointEnv - Setpoint environment with a state consisting of key variables

VisualBallBeamSetpointEnv - Setpoint environment with simulation plot as state
"""
import time
import numpy as np
from gym import spaces
from ballbeam_gym.envs.base import BallBeamBaseEnv, VisualBallBeamBaseEnv

# class Custom_reward():
#     def __init__(self,):
#         pass
#     def custom_reward(setpoint, current_position, L, action):

#         # diff = abs(setpoint-current_position)
#         c_reward = (1.0 - abs(setpoint - current_position)/L)**2
#         # if diff == 0:
#         #     if action == 1:
#         #         c_reward = custom_reward +1
#         #         if self.bb.theta == 0:
#         #             c_reward = custom_reward + 1
#         #     else:
#         #         c_reward = custom_reward - 1

#         return c_reward
    



class BallBeamSetpointEnv(BallBeamBaseEnv):
    """ BallBeamSetpointEnv

    Setpoint environment with a state consisting of key variables

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    max_timesteps : maximum length of an episode, int

    action_mode : action space, str ['continuous', 'discrete']

    setpoint : target position of ball, float (units)
    """

    def __init__(self, timestep=0.1, beam_length=2.0, max_angle=0.2, 
                 init_velocity=0.0, max_timesteps=300, action_mode='continuous', 
                 setpoint=None, Use_Reward_function_advance=False):
                 
        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle,
                  'init_velocity': init_velocity,
                  'max_timesteps': max_timesteps,
                  'action_mode':action_mode,
                  'Use_Reward_function_advance': Use_Reward_function_advance,
                  }

        super().__init__(**kwargs)

        self.logged_actions = []
        self.logged_positions = []
        self.logged_angles = []
        self.logged_rewards = []

        self.Use_Reward_function_advance = Use_Reward_function_advance
        # random setpoint for None values
        if setpoint is None:
            self.setpoint = np.random.random_sample()*beam_length - beam_length/2
            self.random_setpoint = True
        else:
            print(f'Setpoint: {setpoint} Beam_length/2:  {beam_length/2}./n')

            if abs(setpoint) > beam_length/2:
                raise ValueError('Setpoint outside of beam.')
            self.setpoint = setpoint
            self.random_setpoint = False

        #self.action_angle_log = []

        # Define action space for discrete actions
        self.action_space = spaces.Discrete(3)  # Actions: 0, 1, 2
        #self.action_space = spaces.Box(low=-0.2, high=0.2, shape=(1,), dtype=np.float32)

                                # [angle, position, velocity, setpoint]
        self.observation_space = spaces.Box(low=np.array([-max_angle,
                                                          -np.inf,
                                                          -np.inf,
                                                          -beam_length/2]),
                                            high=np.array([max_angle, 
                                                           np.inf, 
                                                           np.inf, 
                                                           beam_length/2]))
        self.max_angle = max_angle

    # def _action_conversion1(self, action, angle, setpoint, position) :
    #     # 0  --> -   
    #     # 1  --> remain same
    #     # 2  --> +
    #     #self.action_angle_log.append((action, angle))
    #     diff = abs(setpoint-position)*1.2
    #     diff = diff * self.max_angle 
    #     angle1 = min(diff, self.max_angle)
    #     action_angles = [-angle1, 0, +angle1]  
    #     final_action = action_angles[action]

    #     angle_deg = np.degrees(final_action)

    #     #print(f'final_action:  {angle_deg} ')
        
    #     return final_action 
    
    def _action_conversion1(self, action, angle, setpoint, position) :
        # 0  --> -   
        # 1  --> remain same
        # 2  --> +
        #self.action_angle_log.append((action, angle))

        action_angles = [angle-0.01, 0, angle+0.01]  
        final_action = action_angles[action]

        angle_deg = np.degrees(final_action)

        #print(f'final_action:  {angle_deg} ')
        
        return final_action 
#..................................................................................................................................................................................
    # def calculate_reward(self, setpoint, position, beam_length, action, velocity):

    #     reward = (1.0 - abs(setpoint - position)/beam_length)**2     

    #     return max(0, min(1, reward))
    



#.....................................................................................................................................................................................

    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/descrease angle, int [0, 1]
        """
        super().step()

        # Action 0 (Increase): Increases the angle.
        # Action 1 (Maintain): Maintains the current angle.
        # Action 2 (Decrease): Decreases the angle.

        self.bb.update(self._action_conversion1(action,self.bb.theta,self.setpoint, self.bb.x ))

        #self.bb.update(self._action_conversion(action))
        obs = np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

        #obs = np.array([self.bb.theta, self.bb.x, 0, self.setpoint])
        #print(action, ', ', self.bb.theta)

        # # Reward function based on proximity to setpoint
        # distance_to_setpoint = abs(self.setpoint - self.bb.x)
        # reward = (1.0 - distance_to_setpoint / self.bb.L) ** 2
#        
        # reward squared proximity to setpoint 
        diff = abs(self.setpoint - self.bb.x)
        #custom_reward = (1.0 - abs(self.setpoint - self.bb.x)/self.bb.L)**2 

        from Simulation.simulation_reward import calculate_reward
        custom_reward = calculate_reward(self.setpoint, self.bb.x, self.bb.L, action, self.bb.v, self.Use_Reward_function_advance)

        # if diff < 0.1:
        #     if action == 1:
        #         custom_reward = custom_reward +1
        #         if self.bb.theta == 0:
        #             custom_reward = custom_reward + 1
        #     else:
        #         custom_reward = custom_reward - 1


        # angle_penalty = abs(self.bb.theta) 
        # reward = proximity_reward - angle_penalty  


        #reward = Custom_reward.custom_reward(self.setpoint, self.bb.x, self.bb.L, action)
        #print(f'position: { self.bb.x},  action:  {action}')
        #time.sleep(0.03)
                # Log values

        self.logged_actions.append(action)
        self.logged_positions.append(self.bb.x)
        self.logged_angles.append(self.bb.theta)
        self.logged_rewards.append(custom_reward)

        return obs, custom_reward, self.done, {}

    def plot_logs(self, save_path):
        """
        Plot the logged data after an episode or training session.
        """
        import matplotlib.pyplot as plt
        import os

        plot_dir = os.path.join(save_path, f"plots")
        os.makedirs(plot_dir, exist_ok=True)

        plt.figure(figsize=(12, 8))

        # Plot rewards
        plt.subplot(4, 1, 1)
        plt.plot(self.logged_rewards, label='Rewards')
        plt.xlabel('Timesteps')
        plt.ylabel('Reward')
        plt.legend()

        # Plot positions
        plt.subplot(4, 1, 2)
        plt.plot(self.logged_positions, label='Positions', color='orange')
        plt.xlabel('Timesteps')
        plt.ylabel('Position')
        plt.legend()

        # Plot angles
        plt.subplot(4, 1, 3)
        plt.plot(self.logged_angles, label='Angles', color='green')
        plt.xlabel('Timesteps')
        plt.ylabel('Angle')
        plt.legend()

        # Plot actions
        plt.subplot(4, 1, 4)
        plt.plot(self.logged_actions, label='Actions', color='purple')
        plt.xlabel('Timesteps')
        plt.ylabel('Action')
        plt.legend()

        plt.tight_layout()
        import os
        # Save the figure as an image in the new folder
        plot_filename = os.path.join(plot_dir, 'progress_plots.png')
        plt.savefig(plot_filename)
        # plt.show()
        self.scatter_plot_metrics(plot_dir)


    def scatter_plot_metrics(self, plot_dir):
        """
        Create scatter plots for Position vs Reward, Position vs Angle, and Position vs Action.
        """
        import matplotlib.pyplot as plt
        from datetime import datetime
        import os

        plt.figure(figsize=(12, 8))

        # Scatter: Position vs. Reward
        plt.subplot(2, 2, 1)
        plt.scatter(self.logged_positions, self.logged_rewards, c='blue', alpha=0.6, label='Position vs Reward')
        plt.xlabel('Position')
        plt.ylabel('Reward')
        plt.title('Position vs Reward')
        plt.legend()

        # Scatter: Position vs. Angle
        plt.subplot(2, 2, 2)
        plt.scatter(self.logged_positions, self.logged_angles, c='orange', alpha=0.6, label='Position vs Angle')
        plt.xlabel('Position')
        plt.ylabel('Angle')
        plt.title('Position vs Angle')
        plt.legend()

        # Scatter: Position vs. Action
        plt.subplot(2, 2, 3)
        plt.scatter(self.logged_positions, self.logged_actions, c='green', alpha=0.6, label='Position vs Action')
        plt.xlabel('Position')
        plt.ylabel('Action')
        plt.title('Position vs Action')
        plt.legend()

        # Scatter: Position vs. Action
        plt.subplot(2, 2, 4)
        plt.scatter(self.logged_positions[-2000:], self.logged_actions[-2000:], c='green', alpha=0.6, label='Position vs Action')
        plt.xlabel('Position')
        plt.ylabel('Action')
        plt.title('Position vs Action last 2000 steps')
        plt.legend()

        plt.tight_layout()
        import os
        # Save the figure as an image in the new folder
        plot_filename = os.path.join(plot_dir, 'Scatter_plots.png')
        plt.savefig(plot_filename)
        # plt.show()

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        super().reset()
        
        if self.random_setpoint is None:
            self.setpoint = np.random.random_sample()*self.beam_length - self.beam_length/2

        return np.array([self.bb.theta, self.bb.x, self.bb.v, self.setpoint])

    # def print_action_angle_table(self):
    #     print("Action | Angle")
    #     print("--------------")
    #     for action, angle in self.action_angle_log:
    #         print(f"{action}     | {angle}")

class VisualBallBeamSetpointEnv(VisualBallBeamBaseEnv):
    """ VisualBallBeamSetpointEnv

    Setpoint environment with simulation plot as state

    Parameters
    ----------
    time_step : time of one simulation step, float (s)

    beam_length : length of beam, float (units)

    max_angle : max of abs(angle), float (rads) 

    init_velocity : initial speed of ball, float (units/s)

    max_timesteps : maximum length of an episode, int

    action_mode : action space, str ['continuous', 'discrete']

    setpoint : target position of ball, float (units)
    """
    
    def __init__(self, timestep=0.1, beam_length=1.0, max_angle=0.2, 
                 init_velocity=0.0, max_timesteps=300, action_mode='continuous', 
                 setpoint=None):

        kwargs = {'timestep': timestep,
                  'beam_length': beam_length,
                  'max_angle':max_angle,
                  'init_velocity': init_velocity,
                  'max_timesteps': max_timesteps,
                  'action_mode':action_mode}

        super().__init__(**kwargs)

        # random setpoint for None values
        if setpoint is None:
            self.setpoint = np.random.random_sample()*beam_length - beam_length/2
            self.random_setpoint = True
        else:
            if abs(setpoint) > beam_length/2:
                raise ValueError('Setpoint outside of beam.')
            self.setpoint = setpoint
            self.random_setpoint = False

        self.max_angle = max_angle
        
    # def _action_conversion1(self, action, angle, setpoint, position) :
    # #     #0 mean --> -0.02  
    # #     #1 mean --> remain same
    # #     #2 mean --> +0.02s
    #     diff = abs(setpoint-position)*1.2
    #     diff = diff * self.max_angle #scaling
    #     angle1 = min(diff, self.max_angle)
    #     action_angles = [-angle1, 0, +angle1]  

    #     return action_angles[action]

    def step(self, action):
        """
        Update environment for one action

        Parameters
        ----------
        action [continuous] : set angle, float (rad)
        action [discrete] : increase/keep/descrease angle, int [0, 1, 2]
        """
        super().step()

        # self.bb.update(self._action_conversion1(action,self.bb.theta,self.setpoint, self.bb.x ))

        self.bb.update(self._action_conversion(action))
        obs = self._get_state()

        # reward squared proximity to setpoint 
        #reward = (1.0 - abs(self.setpoint - self.bb.x)/self.bb.L)**2
        # reward = Custom_reward.custom_reward(self.setpoint, self.bb.x, self.bb.L, action)
        print(f'Setpoint: {self.setpoint}/n')
        diff = abs(self.setpoint - self.bb.x)
        custom_reward = (1.0 - abs(self.setpoint - self.bb.x)/self.bb.L)**2 
        # if diff < 0.1:
        #     if action == 1:
        #         custom_reward = custom_reward +1
        #         if self.bb.theta == 0:
        #             custom_reward = custom_reward + 1
        #     else:
        #         custom_reward = custom_reward - 1


        return obs, custom_reward, self.done, {}

    def reset(self):
        """ 
        Reset environment

        Returns
        -------
        observation : simulation state, np.ndarray (state variables)
        """
        
        if self.random_setpoint is None:
            self.setpoint = np.random.random_sample()*self.beam_length \
                            - self.beam_length/2

        return super().reset()
