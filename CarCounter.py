import cv2
import torch
import numpy as np
import math
from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort
model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)

cap = cv2.VideoCapture('Konya.mp4')

# defining a area
area_1 = np.array([[450, 438], [363, 302], [590, 252], [692, 348]])
line_1 = np.array([[354, 363], [668, 287]])


area1 = set()
area2 = set()

# Global variables to keep track of mouse events and lines
drawing = False
start_point = (-1, -1)
lines = []
lines_coordinates = []  # List to store the coordinates of the lines

def draw_line(event, x, y, flags, param):
    global drawing, start_point, lines, lines_coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        end_point = (x, y)
        lines.append((start_point, end_point))
        lines_coordinates.append((start_point, end_point))  # Store the coordinates of the line
    else:
        POINTS(event, x, y, flags, param)

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)
        
# Equation of the line
def line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - (m * x1)
    return m, c

# Distance from point to line
def distance_point_line(m, c, x2, y2):
    distance = abs(m * x2 - y2 + c) / (math.sqrt((m * m) + 1))
    return distance

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', draw_line)

tracker = Tracker()
while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1020, 500))
    cv2.polylines(frame, [area_1], True, (0, 255, 255), 2)
    cv2.line(frame, line_1[0],line_1[1], (0, 255, 255), 2)

    # Drawing the lines
    for line in lines:
        cv2.line(frame, line[0], line[1], (255, 255, 255), 2)

    # Draw coordinates above the lines
    for idx, line_coord in enumerate(lines_coordinates):
        text = f"{line_coord[0]} {line_coord[1]}"  # Example: "X1 Y1 X2 Y2"
        cv2.putText(frame, text, line_coord[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    results = model(frame)
    list = []
    for index, row in results.pandas().xyxy[0].iterrows():
        x1 = int(row['xmin'])
        y1 = int(row['ymin'])
        x2 = int(row['xmax'])
        y2 = int(row['ymax'])
        label = row['name']
        confidence = row['confidence']
        if 'car' in label:
            list.append([x1, y1, x2, y2])           

    m, c = line_equation(line_1[0][0], line_1[0][1], line_1[1][0], line_1[1][1])         
    # Putting rectangle and text on each person
    boxes_ids = tracker.update(list)
    for box_id in boxes_ids:
        x, y, w, h, id = box_id
        cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 2)
        cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

        # Calculate the bottom center point
        bottom_center_x = int((x + w) / 2)
        bottom_center_y = h

        # Draw a red dot at the bottom center
        cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)

        bot_mid_point = int(x + (w - x) / 2), int(h)
        result = cv2.pointPolygonTest(area_1, (bot_mid_point), False)
        distance = distance_point_line(m, c, bot_mid_point[0], bot_mid_point[1])
        if distance < 30:
            area2.add(id)

        if result > 0:
            area1.add(id)

    p = len(area1)
    l = len(area2)
    
    cv2.putText(frame, ('Person Count In Rectangle=' + str(p)), (10, 470), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    cv2.putText(frame, ('Person Count In Line=' + str(l)), (15, 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 2)
    # Showing the frame
    cv2.imshow('FRAME', frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
