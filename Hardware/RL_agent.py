from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
import sys
import time
# Adding Hardware directory to system path
sys.path.append('./Hardware')
import Hardware_environment 

class RLAgent:
    def __init__(self, env_class, model_save_path='./logs/', checkpoint_freq=500):
        """
        Initialize the RLAgent with environment and model settings.

        :param env_class: The environment class to be used.
        :param model_save_path: Path to save model checkpoints.
        :param checkpoint_freq: Frequency to save model checkpoints.
        """
        self.env = env_class
        check_env(self.env, warn=True)
        self.env = DummyVecEnv([lambda: self.env])

        self.policy_kwargs = dict(
            activation_fn=th.nn.ReLU,
            net_arch=[dict(pi=[256, 256], vf=[256, 256])]
        )

        self.model = None
        self.checkpoint_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=model_save_path)


    def create_new_model(self):
        """
        Create a new PPO model.
        """

        self.model = DQN(
            'MlpPolicy',
            self.env,
            verbose=1,
            learning_rate=1e-4,
            buffer_size=100000,
            batch_size=32,
            gamma=0.99,
            exploration_fraction=0.1,
            exploration_final_eps=0.02
        )
        # self.model = PPO(
        #     policy="MlpPolicy",
        #     env=self.env,
        #     learning_rate=3e-4,
        #     n_steps=500,
        #     batch_size=512,
        #     n_epochs=10,
        #     gamma=0.95,
        #     gae_lambda=0.95,
        #     clip_range=0.2,
        #     policy_kwargs=self.policy_kwargs,
        #     verbose=2,
        # )

    def load_model(self, load_path):
        """
        Load a saved model.

        :param path: Path to the saved model.
        """
        self.model = DQN.load(load_path)
        self.model.set_env(self.env)

    def train_agent(self, get_url, post_url, episode_length, num_steps, Use_pre_trained_agent, load_path, save_path):
        """
        Train the RL agent.

        :param total_timesteps: Total timesteps for training.
        """
        self.create_new_model()
        if Use_pre_trained_agent:
            self.load_model(load_path)

        if self.model is None:
            raise ValueError("Model is not initialized. Create or load a model first.")
        self.model.learn(total_timesteps=num_steps, callback=self.checkpoint_callback)
        self.save_model(save_path)

    def save_model(self, path):
        """
        Save the trained model.

        :param path: Path to save the model.
        """
        if self.model is None:
            raise ValueError("Model is not initialized. Create or load a model first.")
        self.model.save(path)


    def test_agent(self, get_url, post_url, episode_length, num_steps, load_path):
        """
        Test the RL agent.

        :param model_path: Path to the saved model to be loaded for testing. If None, uses the current model.
        :param max_steps: Maximum number of steps for testing.
        """
        if load_path:
            print(f'load_path: {load_path}, num_steps: {num_steps}, episode_length: {episode_length} ')
            self.load_model(load_path)
        else:
            raise ValueError("No model available. Create or load a model first.")

        state = self.env.reset()

        total_reward = 0
        done = False
        steps = 0
    
        while not done and steps < num_steps:
            action, _ = self.model.predict(state)
            state, reward, done, info = self.env.step(action)
            total_reward += reward
            steps += 1
            time.sleep(0.05)
            print(f"Step {steps}: Action={action}, State={state}, Reward={reward}")

        print(f"Total reward: {total_reward}")




