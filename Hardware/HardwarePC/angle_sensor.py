import serial
import time
import requests

serial_port = 'COM5'  
baud_rate = 115200  

url = "http://192.168.1.1:5000/update-sensor"

def read_serial_data(url):
    try:
        ser = serial.Serial(serial_port, baud_rate, timeout=1)
        time.sleep(2)  
        print("Connected to Arduino.")
        
        while True:
            if ser.in_waiting > 0:
                angle = ser.readline().decode('utf-8').strip()  
                data = {"angle_current": angle}
                response = requests.post(url, json=data)
                if response.status_code == 200:
                    print(f"Updated angle: {angle}")

    except serial.SerialException as e:
        print(f"Error connecting to the serial port: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    print("Starting to read serial data from Arduino...")
    read_serial_data(url)

if __name__ == "__main__":
    main()
