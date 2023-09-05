import cv2

# Open the video file
video_path = 'C:\\Users\\Berk\\Videos\\Captures\\v17.mp4'
cap = cv2.VideoCapture(video_path)
i = 38159
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties
frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
capture_interval = int(frame_rate * 5)  # Capture every 5 seconds

# Create a directory to save screenshots
import os
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

# Capture and save screenshots
current_frame = 0

while current_frame < total_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    if current_frame % capture_interval == 0:
        screenshot_path = f"screenshots/screenshot_{i}.jpg"
        print(f"Saving screenshot at frame {i} to {screenshot_path}")
        cv2.imwrite(screenshot_path, frame)
    i += 1
    current_frame += 1

# Release video capture object
cap.release()
cv2.destroyAllWindows()

print("Screenshots saved successfully!")
