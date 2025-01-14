import requests
import time
import matplotlib.pyplot as plt

def run_pid_control(K_p, K_i, K_d, post_url, get_url):
    set_angle_list = []
    get_angle_list = []
    time_taken_list = []
    ball_position_list = []
    ball_speed_list = []
    
    integral = 0  # Integral accumulator
    previous_position = None  # To calculate derivative

    def calculate_set_angle(ball_position, speed, dt):
        nonlocal integral, previous_position
        
        # Update integral
        integral += ball_position * dt

        # Calculate derivative
        derivative = 0 if previous_position is None else (ball_position - previous_position) / dt
        previous_position = ball_position

        # PID control equation
        set_angle = (K_p * ball_position) + (K_i * integral) + (K_d * derivative)
        return max(-15, min(15, set_angle))

    try:
        while True:
            start_time = time.time()

            get_response = requests.get(get_url)
            if get_response.status_code == 200:
                data = get_response.json()
                print(f'Data received: {data}')

                ball_position = data.get('position')
                speed = data.get('speed')
                get_angle = data.get('angle_current')

                ball_position_list.append(ball_position)
                ball_speed_list.append(speed)
                get_angle_list.append(get_angle)

                if ball_position is not None and speed is not None:
                    # Time delta for integral and derivative calculations
                    dt = time_taken_list[-1] if time_taken_list else 0.05
                    set_angle = calculate_set_angle(ball_position, speed, dt)
                else:
                    set_angle = 0  # Default to zero if data is invalid

                set_angle_list.append(set_angle)

                payload1 = {"angle_set": set_angle}
                response = requests.post(post_url, json=payload1)

                if response.status_code == 200:
                    print(f"Set angle updated successfully! Angle: {set_angle}")

            time.sleep(0.05)  # 50 ms delay

            end_time = time.time()
            time_taken = end_time - start_time
            time_taken_list.append(time_taken)

            print(f'Time Taken: {time_taken}')

    except KeyboardInterrupt:
        print("Loop interrupted. Plotting the results...")

        plt.figure(figsize=(12, 10))

        plt.subplot(4, 1, 1)
        plt.plot(set_angle_list, label='Set Angle')
        plt.plot(get_angle_list, label='Get Angle')
        plt.title('Set Angle and Get Angle Comparison')
        plt.xlabel('Iteration')
        plt.ylabel('Angle')
        plt.legend()

        plt.subplot(4, 1, 2)
        plt.plot(ball_position_list, label='Ball Position', color='orange')
        plt.title('Ball Position Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Position')
        plt.legend()

        plt.subplot(4, 1, 3)
        plt.plot(ball_speed_list, label='Ball Speed', color='purple')
        plt.title('Ball Speed Over Time')
        plt.xlabel('Iteration')
        plt.ylabel('Speed')
        plt.legend()

        plt.subplot(4, 1, 4)
        plt.plot(time_taken_list, label='Time Taken', color='green')
        plt.title('Time Taken for Each Request Cycle')
        plt.xlabel('Iteration')
        plt.ylabel('Time (s)')
        plt.legend()

        plt.tight_layout()
        plt.show()
