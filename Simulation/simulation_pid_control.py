import gym
import ballbeam_gym
import requests
import time
import matplotlib.pyplot as plt

def run_pid_control(K_p, K_i, K_d, setpoint, num_steps):
    # pass env arguments as kwargs
    kwargs = {'timestep': 0.05, 
            'setpoint': setpoint,
            'beam_length': 2.0,
            'max_angle': 0.2,
            'init_velocity': 0.0}

    # create env
    env = gym.make('BallBeamSetpoint-v0', **kwargs)
    #env = DummyVecEnv([lambda: env1])

    #print(env)
    env.reset()

    print("Environment ready..../n")
    print(f'Setpoint: {setpoint} /n')
    # simulate 1000 steps
    for i in range(num_steps):   
        # control theta with a PID controller
        env.render()
        print(f'error: {env.bb.x - env.setpoint} ')
        theta = K_p*(env.bb.x - env.setpoint) + K_d*(env.bb.v)


        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(round(theta))  # Convert theta to an integer index if action_space is Discrete
            action = max(0, min(action, env.action_space.n - 1))  # Clip to valid range
        else:
            action = theta  # If action space is continuous, use theta as is

        obs, reward, done, info = env.step(action)

        if done:
            env.reset()

