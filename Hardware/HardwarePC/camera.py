import cv2
import time
import requests

#x1,y1 = 100, 250
#x2,y2 = 580, 275

#x1,y1 = 65, 250
#x2,y2 = 570, 275


url = "http://192.168.1.1:5000/update-sensor"

def camera_setup(camera):
    # Set the resolution (width and height)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    camera.set(cv2.CAP_PROP_FPS, 15)

def capture_image(camera):
    ret, frame = camera.read()
    if ret:
        return frame
    else:
        print("Failed to capture image")
        return None

# Function to check if the ROI coordinates are valid
def validate_roi(x1, y1, x2, y2):
    return x1 < x2 and y1 < y2

# Function to show the current ROI and wait for user confirmation
def calibrate_roi(camera):
    # Give the camera time to warm up
    print("Warming up the camera for calibration...")
    for _ in range(5):  # Capture a few frames to allow camera to stabilize
        frame = capture_image(camera)
        time.sleep(0.05)  # Small delay between frames

    # Capture a single frame for ROI calibration
    frame = capture_image(camera)
    if frame is None:
        print("Error capturing frame for calibration.")
        return None

    while True:
        # Get user input for the ROI
        try:
            print("Enter the coordinates of the ROI (x1, y1, x2, y2): ")
            x1 = int(input("x1 (Top-Left X): "))
            y1 = int(input("y1 (Top-Left Y): "))
            x2 = int(input("x2 (Bottom-Right X): "))
            y2 = int(input("y2 (Bottom-Right Y): "))
        except ValueError:
            print("Invalid input. Please enter valid integer values.")
            continue

        # Validate the ROI coordinates
        if not validate_roi(x1, y1, x2, y2):
            print("Invalid ROI coordinates. Top-left must be above and to the left of bottom-right.")
            continue

        # Create a copy of the frame for drawing the ROI
        roi_frame = frame.copy()

        # Draw a red rectangle around the ROI
        cv2.rectangle(roi_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Calculate and draw the center of the ROI with a BGR (255, 0, 255) magenta point
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.circle(roi_frame, (center_x, center_y), 5, (255, 0, 255), -1)

        # Display the frame with the ROI
        cv2.imshow("Calibrated ROI", roi_frame)
        cv2.waitKey(500)  # Display for 500 ms

        # Ask user if the ROI is correct
        key = input("Is this ROI correct? (y/n): ").strip().lower()
        if key == 'y':
            cv2.destroyWindow("Calibrated ROI")
            return x1, y1, x2, y2
        else:
            print("Let's recalibrate the ROI.")
            frame = capture_image(camera)  # Re-capture the frame if ROI is incorrect


# Function to detect balls in the frame
def detect_balls_in_roi(frame, roi_coords, detect_black_ball=True):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    black_ball_position = None
    steel_ball_position = None

    if detect_black_ball:
        _, black_mask = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY_INV)
        black_contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if black_contours:
            largest_black_contour = max(black_contours, key=cv2.contourArea)
            (bx, by, bw, bh) = cv2.boundingRect(largest_black_contour)
            black_ball_position = (bx + bw // 2, by + bh // 2)
            cv2.drawContours(roi, [largest_black_contour], -1, (0, 255, 0), 2)
            cv2.circle(roi, black_ball_position, 5, (0, 255, 0), -1)

    if not detect_black_ball:
        _, steel_mask = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
        steel_contours, _ = cv2.findContours(steel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if steel_contours:
            largest_steel_contour = max(steel_contours, key=cv2.contourArea)
            (sx, sy, sw, sh) = cv2.boundingRect(largest_steel_contour)
            steel_ball_position = (sx + sw // 2, sy + sh // 2)
            cv2.drawContours(roi, [largest_steel_contour], -1, (255, 0, 0), 2)
            cv2.circle(roi, steel_ball_position, 5, (255, 0, 0), -1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 255), -1)

    if black_ball_position:
        black_ball_position = (black_ball_position[0] + x1, black_ball_position[1] + y1)
    if steel_ball_position:
        steel_ball_position = (steel_ball_position[0] + x1, steel_ball_position[1] + y1)

    return black_ball_position, steel_ball_position

# Function to calculate the speed and relative position of the ball
def calculate_speed_and_direction(pos1, pos2, time_interval, roi_width, roi_center_x):
    if pos1 is None or pos2 is None:
        return None, None, None

    displacement = pos2[0] - pos1[0]
    speed = displacement / time_interval
    direction = "Right" if displacement > 0 else "Left"
    relative_position = (pos2[0] - roi_center_x) / (roi_width // 2)

    return speed, direction, relative_position

# Function to update speed and position to the server
def update_to_server(url, speed, position):
    data = {
        "speed": speed,
        "position": position
    }
    response = requests.post(url, json=data)

    if response.status_code == 200:
        print("Camera sensor data updated successfully")
    else:
        print("Failed to update camera sensor data")


def main():
    # Ask user to choose the camera
    camera_choice = input("Which camera do you want to use? (0: Laptop Webcam, 1: USB Webcam): ")
    camera_index = 0 if camera_choice == '0' else 1

    # Initialize the selected camera
    camera = cv2.VideoCapture(camera_index)
    camera_setup(camera)

    # Calibrate the ROI
    print("Please calibrate the ROI.")
    roi_coords = calibrate_roi(camera)
    roi_width = roi_coords[2] - roi_coords[0]
    roi_center_x = (roi_coords[0] + roi_coords[2]) // 2

    # Ask user which ball to detect
    ball_choice = input("Which ball do you want to detect? (1: Black Rubber Ball, 2: Steel Ball): ")
    detect_black_ball = True if ball_choice == '1' else False

    prev_position = None
    time_interval = 0.1  # 100 ms in seconds, 10 FPS

    last_update_time = time.time()

    while True:
        frame = capture_image(camera)
        if frame is None:
            break

        black_pos, steel_pos = detect_balls_in_roi(frame, roi_coords, detect_black_ball)

        current_time = time.time()  # Get the current time at the start of the loop

        if detect_black_ball and prev_position is not None and black_pos is not None:
            speed, direction, relative_position = calculate_speed_and_direction(prev_position, black_pos, time_interval, roi_width, roi_center_x)
            print(f"Black Ball Speed: {speed:.2f} pixels/s, Direction: {direction}, Relative Position: {relative_position:.2f}")
        if current_time - last_update_time >= 0.3:  # Check if 300 ms have passed
            update_to_server(url, speed, relative_position)
            last_update_time = current_time  # Reset last update time

        if not detect_black_ball and prev_position is not None and steel_pos is not None:
            speed, direction, relative_position = calculate_speed_and_direction(prev_position, steel_pos, time_interval, roi_width, roi_center_x)
            print(f"Steel Ball Speed: {speed:.2f} pixels/s, Direction: {direction}, Relative Position: {relative_position:.2f}")
        if current_time - last_update_time >= 0.3:  # Check if 300 ms have passed
            update_to_server(url, speed, relative_position)
            last_update_time = current_time  # Reset last update time

        prev_position = black_pos if detect_black_ball else steel_pos

        cv2.imshow("Detected Ball", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
