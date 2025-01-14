import serial
import time
import requests
import random
# Settings
com_port = 'COM10'  # Change this to the correct COM port
baud_rate = 115200
server_url_update = "http://192.168.1.1:5000/update-sensor"  # Change <server_ip> to the server's IP
server_url_get = "http://192.168.1.1:5000/get-data"  # Change <server_ip> to the server's IP
request_interval = 0.2  # Interval to request angle in seconds

def update_webserver(angle):
    payload = {"angle_current": angle}
    try:
        response = requests.post(server_url_update, json=payload)
        if response.status_code == 200:
            print("Sensor data updated successfully!")
        else:
            print(f"Failed to update sensor data: {response.status_code}")
    except Exception as e:
        print(f"Exception occurred: {e}")

def get_setpoint_from_webserver():
    try:
        response = requests.get(server_url_get)
        if response.status_code == 200:
            data = response.json()
            return data["angle_set"]
        else:
            print(f"Failed to get setpoint: {response.status_code}")
            return None
    except Exception as e:
        print(f"Exception occurred: {e}")
        return None

def main():
    # Open Serial Connection
    ser = serial.Serial(com_port, baud_rate)
    time.sleep(2)  # Wait for the serial connection to establish

    # Send 'y' to start the calibration process
    while True:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            print(line)
            if "Please enter 'y' to start the calibration process:" in line:
                ser.write(b'y\n')
                break

    # Print messages until calibration is finished
    calibration_finished = False
    while not calibration_finished:
        if ser.in_waiting > 0:
            line = ser.readline().decode().strip()
            print(line)
            if "Calibration finished." in line:
                calibration_finished = True

    previous_setpoint = None  # To track the last setpoint sent

    # Main communication loop
    while True:
        r_num = random.uniform(0.2, 0.22) 

        # # Request current angle from Arduino
        # ser.write(b'GET_ANGLE\n')

        # # Read the response
        # if ser.in_waiting > 0:
        #     line = ser.readline().decode().strip()
        #     if "Current Angle:" in line:
        #         angle = float(line.split(":")[1].strip())
        #         print(f"Current Angle: {angle}")  # Print the angle to the console

        #         # Update the webserver with the current angle
        #         #update_webserver(angle)

        #         # Get the setpoint from the webserver
        #         setpoint = get_setpoint_from_webserver()
        #         if setpoint is not None and setpoint != previous_setpoint:
        #             # Send the new setpoint to the Arduino only if it has changed
        #             ser.write(f"{setpoint}\n".encode())
        #             print(f"Sent Setpoint: {setpoint}")
        #             previous_setpoint = setpoint  # Update the last sent setpoint

        #         # if setpoint is not None:
        #         #     # Send the setpoint to the Arduino
        #         #     ser.write(f"{setpoint}\n".encode())
        #         #     print(f"Sent Setpoint: {setpoint}")
        #         # Get the setpoint from the webserver

        setpoint = get_setpoint_from_webserver()
        #if setpoint is not None and setpoint != previous_setpoint:
            # Send the new setpoint to the Arduino only if it has changed
        print(f"Sent Setpoint_before send: {setpoint}")
        start_time = time.time()
        ser.write(f"{setpoint}\n".encode())
        end_time = time.time()
        print(f'Time taken: {end_time - start_time}')
        #previous_setpoint = setpoint  # Update the last sent setpoint

        # Wait for the next request
        time.sleep(r_num)

if __name__ == "__main__":
    main()