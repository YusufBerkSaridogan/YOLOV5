import cv2
import numpy as np
import os
#Your path here
video_path = '638229488.mp4'

cap = cv2.VideoCapture(video_path)

# Callback function to get mouse cursor coordinates
def get_mouse_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked at coordinates: ({x}, {y})")

# Set the mouse callback
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', get_mouse_coordinates)

def rescale(frame, scale=0.5):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)

while True:
    # Read the next frame from the video
    ret, frame = cap.read()
        
    frame_resize = rescale(frame)
    cv2.imshow('frame', frame_resize)
        
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
