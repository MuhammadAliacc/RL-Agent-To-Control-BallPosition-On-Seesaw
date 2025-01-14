import requests
import time
import gym
import sys
import socket

class ServerClient:
    def __init__(self, page_server_url_get, page_server_url_post):
        self.page_server_url_get = page_server_url_get
        self.page_server_url_post = page_server_url_post

        self.shared_data = {}

    def fetch_data(self):
        """
        Fetch data from the server and store it in shared_data.
        """
        try:
            response = requests.get(self.page_server_url_get)
            if response.status_code == 200:
                self.shared_data = response.json()
                print(f"printing data after fetch TM: {self.shared_data} \n\n")
                return self.shared_data
            else:
                print(f"Failed to fetch data. HTTP Status Code: {response.status_code}")
                return None
        except requests.exceptions.RequestException as e:
            print(f"Error connecting to the server: {e}")
            return None
        
    def reset_variables_on_server(self, page_server_url_post):

        print("Reset variables function called")
        print(page_server_url_post)
        print(' ')
        payload = {"train_new_model_from_simulation": False,
                    "testing_trained_model_on_simulation": False,
                    "Train_RL": False,
                    "Test_RL": False,
                    "PID": False
                    }  
        try:
            response = requests.post(page_server_url_post, json=payload)
            if response.status_code == 200:
                print("Successfully reset variables on server.")
            else:
                print(f"Failed to update. HTTP Status Code: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Error sending request to the server: {e}")
        

    def get_variables(self):
        """
        Retrieve and return variables from the shared data.
        """
        if not self.shared_data:
            print("No data available. Fetch data first.")
            return None

        return {
            "PID": self.shared_data.get("PID", False),
            "Train_RL": self.shared_data.get("Train_RL", False),
            "Test_RL": self.shared_data.get("Test_RL", False),
            "testing_trained_model_on_simulation": self.shared_data.get("testing_trained_model_on_simulation", False),
            "train_new_model_from_simulation": self.shared_data.get("train_new_model_from_simulation", False),
            "agent_name_to_save_from_hardware": self.shared_data.get("agent_name_to_save_from_hardware", False),
            "agent_name_to_load_for_hardware": self.shared_data.get("agent_name_to_load_for_hardware", False),
            "agent_name_to_save_from_simulation": self.shared_data.get("agent_name_to_save_from_simulation", False),
            "agent_name_to_load_for_simulation": self.shared_data.get("agent_name_to_load_for_simulation", False),
            "pre_trained_agent_name_for_simulation": self.shared_data.get("pre_trained_agent_name_for_simulation", False),
            "pre_trained_agent_name_for_hardware": self.shared_data.get("pre_trained_agent_name_for_hardware", False),
            "render": self.shared_data.get("render", False),
            "start": self.shared_data.get("start", False),
            "stop": self.shared_data.get("stop", False),
            "Use_Reward_function_advance": self.shared_data.get("Use_Reward_function_advance", False),
            "Use_pre_trained_agent": self.shared_data.get("Use_pre_trained_agent", False),
            "run_pid_on_hardware": self.shared_data.get("run_pid_on_hardware", False),
            "Episode_Length": self.shared_data.get("Episode_Length", 0),
            "num_steps": self.shared_data.get("num_steps", 0),
            "Setpoint": self.shared_data.get("Setpoint", None),
            "Set_Kp": self.shared_data.get("Set_Kp", 0.0),
            "Set_Ki": self.shared_data.get("Set_Ki", 0.0),
            "Set_Kd": self.shared_data.get("Set_Kd", 0.0),
            "tolerance": self.shared_data.get("tolerance", 0.0),
        }


class TaskManager:
    def __init__(self, server_client):
        self.server_client = server_client

    def run(self):
        """
        Main loop to fetch data and execute tasks based on shared data.
        """
        while True:
            # Fetch the latest data from the server
            shared_data = self.server_client.fetch_data()
            if shared_data:
                variables = self.server_client.get_variables()
                self.variables = variables
                print(f'printing from task manager:  {variables} /n')
                # Check conditions and execute tasks
                if variables["Train_RL"]:
                    self.train_rl(get_url, post_url, variables["Episode_Length"], variables["num_steps"],  Use_pre_trained_agent = variables['Use_pre_trained_agent'], load_path= variables['pre_trained_agent_name_for_simulation'], save_path=variables["agent_name_to_save_from_hardware"], tolerance=variables["tolerance"])
                elif variables["Test_RL"]:
                    self.test_rl(get_url, post_url, variables["Episode_Length"], variables["num_steps"], variables["agent_name_to_load_for_hardware"], tolerance=variables["tolerance"])
                elif variables["train_new_model_from_simulation"]:
                    self.train_agent_from_simulation(variables["Episode_Length"], variables["Use_Reward_function_advance"], variables["Setpoint"],  Use_pre_trained_agent = variables['Use_pre_trained_agent'], load_path= variables['pre_trained_agent_name_for_simulation'], save_path= variables['agent_name_to_save_from_simulation'])                        
                elif variables["testing_trained_model_on_simulation"]:
                    self.test_agent_on_simulation(variables["num_steps"], variables["Setpoint"], variables["agent_name_to_load_for_simulation"])
                elif variables["PID"]: 
                    self.run_pid_control(variables["Set_Kp"], variables["Set_Ki"], variables["Set_Kd"], variables["Setpoint"], variables["num_steps"], variables["run_pid_on_hardware"])

            time.sleep(2)  # Polling interval

    def train_rl(self, get_url, post_url, episode_length, num_steps, Use_pre_trained_agent, load_path, save_path, tolerance):
        """
        Train the RL agent using the given URLs for environment communication.
        :param get_url: URL to get data from the server.
        :param post_url: URL to post data to the server.
        :param episode_length: Length of each training episode.
        :param num_steps: Number of steps to train the agent.
        """
        self.server_client.reset_variables_on_server(page_server_reset_variables)

        from Hardware.RL_agent import RLAgent
        from Hardware import Hardware_environment
        RLAgent= RLAgent(env_class=Hardware_environment.HardwareEnv(get_url=get_url, post_url=post_url, max_steps=num_steps, tolerance=tolerance))
        save_path = f'./Models/{save_path}'
        load_path = f'./Models/{load_path}'

        RLAgent.train_agent(get_url, post_url, episode_length, num_steps, Use_pre_trained_agent, load_path, save_path)
    

    def test_rl(self, get_url, post_url, episode_length, num_steps, load_path, tolerance):
        self.server_client.reset_variables_on_server(page_server_reset_variables)
        """
        Simulates testing a reinforcement learning model.
        """
        from Hardware.RL_agent import RLAgent
        from Hardware import Hardware_environment
        RLAgent= RLAgent(env_class=Hardware_environment.HardwareEnv(get_url=get_url, post_url=post_url, max_steps=num_steps, tolerance=tolerance, testing_agent=True))

        load_path = f'./Models/{load_path}'
        RLAgent.test_agent(get_url, post_url, episode_length, num_steps, load_path)
        print("Simulation complete!")


    def run_pid_control(self, K_p, K_i, K_d, setpoint, num_steps, run_pid_on_hardware):
        self.server_client.reset_variables_on_server(page_server_reset_variables)
        """
        Simulates PID control with provided gains.
        """
        if run_pid_on_hardware:
            sys.path.append('./Hardware')
            import Hardware_pid_control
            print(f"Running PID control with Kp={K_p}, Ki={K_i}, Kd={K_d}")
            Hardware_pid_control.run_pid_control(K_p, K_i, K_d, post_url, get_url)
        else:
            sys.path.append('./Simulation')
            import simulation_pid_control
            print(f"Running PID control with Kp={K_p}, Ki={K_i}, Kd={K_d}")
            simulation_pid_control.run_pid_control(K_p, K_i, K_d, setpoint, num_steps)
            pass

    def train_agent_from_simulation(self, episode_length, Use_Reward_function_advance, Setpoint,  Use_pre_trained_agent, load_path, save_path):
        self.server_client.reset_variables_on_server(page_server_reset_variables)
        from Simulation.Simulation_RL_agent_S3 import train_simulation_rl_agent
        print(f'Save model name is = {save_path}')
        save_path = f'./Models/{save_path}'
        load_path = f'./Models/{load_path}'
        train_simulation_rl_agent(
            num_steps = self.variables["num_steps"], 
            episode_length= episode_length, 
            render=self.variables["render"], 
            Use_Reward_function_advance=Use_Reward_function_advance, 
            Setpoint=Setpoint, Use_pre_trained_agent=Use_pre_trained_agent, 
            load_path=load_path, 
            save_path=save_path
            )

    def test_agent_on_simulation(self, num_steps, Setpoint, load_path):
        self.server_client.reset_variables_on_server(page_server_reset_variables)
        from Simulation.Simulation_RL_agent_S3 import test_simulation_rl_agent
        load_path = f'./Models/{load_path}'
        test_simulation_rl_agent(render=self.variables["render"], num_steps=self.variables["num_steps"], Setpoint=Setpoint, load_path=load_path)


if __name__ == "__main__":
    
    host =sys.argv[1]

    hardware_server = "169.254.94.35"
    print(f'Hardware selected IP: {hardware_server}')
    # Hardware server
    post_url = f"http://{hardware_server}:5002/set-angle"
    get_url = f"http://{hardware_server}:5002/get-data"
    # Page server 
    page_server_url_post = f"http://{host}:5000/update"
    page_server_url_get = f"http://{host}:5000/get_data"
    page_server_reset_variables = f"http://{host}:5000/reset_variables"


    # Initialize the ServerClient with the server URL
    client = ServerClient(page_server_url_get, page_server_url_post)

    # Initialize the TaskManager with the ServerClient
    task_manager = TaskManager(client)

    # Run the task manager to handle tasks based on fetched data
    task_manager.run()