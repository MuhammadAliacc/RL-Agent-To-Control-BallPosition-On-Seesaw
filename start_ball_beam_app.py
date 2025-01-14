from flask import Flask, request, jsonify, render_template, Response
from waitress import serve
import subprocess
import os
import sys
import signal
import atexit
import time
import threading
import socket
import psutil

app = Flask(__name__)
script_process = None

shared_data = {
    "PID": False,  # bool
    "Train_RL": False,  # bool
    "Test_RL": False,  # bool
    "Use_Reward_function_advance": False,  # bool
    "run_pid_on_hardware": False,  # bool
    "Use_pre_trained_agent": False,
    "testing_trained_model_on_simulation": False,  # bool
    "agent_name_to_save_from_hardware": '',
    "agent_name_to_load_for_hardware": '',
    "agent_name_to_save_from_simulation": '',
    "agent_name_to_load_for_simulation": '',
    "pre_trained_agent_name_for_hardware": '',
    "pre_trained_agent_name_for_simulation": '',
    "train_new_model_from_simulation": False,  # bool
    "render": False,  # bool
    "Episode_Length": 100,  # int
    "num_steps": 100,  # int
    "Setpoint": None,  # int
    "Set_Kp": 1.0,  # float
    "Set_Ki": 0.1,  # float
    "Set_Kd": 0.01,  # float
    "tolerance": 0.01,  # float
    "start": False,  # bool
    "stop": False,  # bool
}

interface_name = "Ethernet 2"

def get_specific_ip(interface_name):
    for interface, addrs in psutil.net_if_addrs().items():
        if interface == interface_name:
            for addr in addrs:
                if addr.family == socket.AF_INET:  
                    return addr.address
    return None
host = get_specific_ip(interface_name)

@app.route('/')
def index():
    host_ip = get_specific_ip(interface_name)
    return render_template('Web_page_html.html', data=shared_data, host_ip=host_ip)

# def start_hardware_server(host, port):
#     try:
#         hardware_server_dir = os.path.join(os.getcwd(), 'Hardware')
#         if not os.path.exists(hardware_server_dir):
#             print(f"Error: Directory '{hardware_server_dir}' does not exist.")
#             return
#         os.chdir(hardware_server_dir)
#         command = [sys.executable, 'hardware_server.py', host, str(port)]
#         subprocess.Popen(command)
#         print(f'Hardware server started at http://{host}:{port}...')
#         os.chdir('..')  
#     except Exception as e:
#         print(f"Error starting hardware server: {str(e)}")


def convert_value(key, value):
    """Convert the value to the correct type based on the key."""
    #if key == "Setpoint":  # Special case for setpoint
        #print(f"Setpoint before convert value: {value}")

    if isinstance(shared_data[key], bool):
        return value.lower() == "true" if isinstance(value, str) else bool(value)
    elif isinstance(shared_data[key], int):
        try:
            return int(value)
        except ValueError:
            return shared_data[key]  
    elif isinstance(shared_data[key], float):
        try:
            if key == "Setpoint":  # Special case for setpoint
                print(f"Setpoint after convert float: {value}")
            return float(value)
        except ValueError:
            if key == "Setpoint":  # Special case for setpoint
                print(f"Setpoint after convert not float: {value}")
            return shared_data[key]  
    else:
        #if key == "Setpoint":  # Special case for setpoint
            #print(f"Setpoint after convert else: {value}")
        return value
    

@app.route('/update', methods=['POST'])
def update():
    global shared_data
    new_data = request.json
    if script_process is None or script_process.poll() is not None:  
        for key, value in new_data.items():
            if key in shared_data:
                shared_data[key] = convert_value(key, value)
        return jsonify({"status": "success", "data": shared_data}), 200
    else:
        return jsonify({'message': 'Error in updating on server.'}), 400

@app.route('/reset_variables', methods=['POST'])
def reset_variables():
    global shared_data
    new_data = request.json
    for key, value in new_data.items():
        if key in shared_data:
            shared_data[key] = convert_value(key, value)
    return jsonify({"status": "success", "data": shared_data}), 200


@app.route('/get_data', methods=['GET'])
def get_data():
    return jsonify(shared_data)

@app.route('/start-python', methods=['POST'])
def start_script():
    """Start the Python script"""
    global script_process
    if script_process is None or script_process.poll() is not None:  
        script_process = subprocess.Popen([sys.executable, 'Task_Manager.py', host])
        print('Main process started ... ')
        return jsonify({'message': 'Script started successfully.'}), 200
    else:
        return jsonify({'message': 'Script is already running.'}), 400


@app.route('/stop-python', methods=['POST'])
def stop_script():
    """Stop the running Python script"""
    global script_process
    if script_process is not None:
        script_process.terminate()  
        script_process = None
        print('Main process stoped ...')
        return jsonify({'message': 'Script stopped successfully.'}), 200
    else:
        return jsonify({'message': 'No script is running.'}), 400


@app.route('/restart-python', methods=['POST'])
def restart_script():
    """Restart the running Python script"""
    global script_process
    try:
        if script_process is not None:
            script_process.terminate()
            script_process.wait()  
            script_process = None

        script_process = subprocess.Popen([sys.executable, 'Task_Manager.py'])
        return jsonify({'message': 'Script restarted successfully.'}), 200
    except Exception as e:
        return jsonify({'message': f'Error restarting script: {str(e)}'}), 500


@app.route('/restart-server', methods=['POST'])
def restart_server():
    """Restart the running server"""
    python = sys.executable  
    subprocess.Popen([python] + sys.argv)  
    os._exit(0)  
    return jsonify({'message': 'Server restarting...'}), 200

def shutdown_server():
    print("Shutting down server...")
    os.kill(os.getpid(), signal.SIGTERM)  
atexit.register(shutdown_server)


# -------------------- Setting up server ---> 
def run_server(host, port):
    print('Starting webpage Server')
    url = f'http://{host}:{port}/'
    print(f'Web page URL: {url}')
    app.debug = True
    serve(app, host=host, port=port)

def open_browser(host, webpage_port):
    url = f'http://{host}:{webpage_port}/'
    chrome_path = "C:\Program Files\Internet Explorer\iexplore.exe"  
    time.sleep(3)
    try:
        subprocess.Popen([chrome_path, url])
        print('Browser opened')
    except Exception as e:
        print(f"Error occurred when opening browser: {e}")

if __name__ == '__main__':
    
    host = get_specific_ip(interface_name)

    webpage_port = 5000
    #hardware_port = 5002

    #start_hardware_server(host, hardware_port)

    server_thread = threading.Thread(target=run_server, args=(host, webpage_port))
    server_thread.start()

    open_browser(host, webpage_port)
    print('Server Started and Browser Opened')