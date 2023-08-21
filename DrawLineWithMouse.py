import cv2
import numpy as np

# Global variables to keep track of mouse events
drawing = False
start_point = (-1, -1)
line_count = 0
text = 'th Line'

def draw_line(event, x, y, flags, param):
    global drawing, start_point, line_count

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        line_count += 1
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        cv2.line(img, start_point, end_point, (255, 255, 255), 2)
        
        cv2.putText(img, str(line_count) + text, (start_point), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.circle(img, start_point, 5, (0, 0, 255), -1)  # Red circle at the start point
        cv2.circle(img, end_point, 5, (0, 0, 255), -1)  # Red circle at the end point,
        cv2.imshow('Image', img)

# Create a black image
width, height = 640, 480
img = np.zeros((height, width, 3), np.uint8)

cv2.imshow('Image', img)
cv2.setMouseCallback('Image', draw_line)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # Press 'Esc' to exit
        break

cv2.destroyAllWindows()
