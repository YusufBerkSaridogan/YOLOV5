import cv2
import torch
import numpy as np
import math
from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    
video_path = 'your_video_path'
cap = cv2.VideoCapture('638229488.mp4')

# defining a area
area_1 = np.array([[450, 438], [363, 302], [590, 252], [692, 348]])
line_1 = np.array([[354, 363], [668, 287]])


areas = {}

# Global variables to keep track of mouse events and lines
drawing = False
start_point = (-1, -1)
lines = []
lines_coordinates = []  # List to store the coordinates of the lines

# Functions Section 
def create_area_variables(quantity):
    areas = {}
    for i in range(1, quantity + 1):
        variable_name = f'area{i}'
        areas[variable_name] = set()
    return areas

def capture_single_frame(video_path, target_size):
    # Load the video
    cap = cv2.VideoCapture(video_path)

    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return None
    
    # Capture the first frame
    ret, frame = cap.read()
    if ret:
        # Release the video capture object
        cap.release()
        
        # Resize the captured frame
        resized_frame = cv2.resize(frame, target_size)
        return resized_frame
    else:
        print("Error: Unable to capture frame.")
        cap.release()
        return None

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
       # print(colorsBGR)
        
# Equation of the line
def line_equation(x1, y1, x2, y2):
    m = (y2 - y1) / (x2 - x1)
    c = y1 - (m * x1)
    return m, c

# Distance from point to line
def distance_point_line(m, c, x2, y2):
    distance = abs(m * x2 - y2 + c) / (math.sqrt((m * m) + 1))
    return distance
#Functions Section End

# Set the output filename
output_filename = 'captured_frame.jpg'

# Call the function to capture a single frame
image = capture_single_frame(video_path, (1020, 500))
show_image = True

cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_line)


while True:
    if show_image:      
        # Drawing the lines
        for line in lines:
            cv2.line(image, line[0], line[1], (255, 255, 255), 2)

        # Draw coordinates above the lines
        for idx, line_coord in enumerate(lines_coordinates):
            text = f"{line_coord[0]} {line_coord[1]}"  # Example: "X1 Y1 X2 Y2"
            cv2.putText(image, text, line_coord[0], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.imshow('Image', image)
        key = cv2.waitKey(1)
        if key == ord('q'):
            show_image = False
            cv2.destroyWindow('Image')
            break
print("Line Coordinates",lines_coordinates)
print("Length of Line Coordinates",len(lines_coordinates))

line_count = len(lines_coordinates)
areas = create_area_variables(line_count)
line_equations = []

for start_point, end_point in lines_coordinates:
    print("Start Point:", start_point)
    print("End Point:", end_point)
    m,c = line_equation(start_point[0], start_point[1], end_point[0], end_point[1])
    line_equations.append((m, c))
print("----------------------------------Loop is Finished----------------------------------")

tracker = Tracker()
while True:
    
    if True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
#       cv2.polylines(frame, [area_1], True, (0, 255, 255), 2)
#       cv2.line(frame, line_1[0],line_1[1], (255, 0, 255), 2)

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
            if 'person' in label:
               list.append([x1, y1, x2, y2])           

      
               
        # Putting rectangle and text on each person
        boxes_ids = tracker.update(list)
        for box_id in boxes_ids:
            x, y, w, h, id = box_id
            cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 1)
            cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

            # Calculate the bottom center point
            bottom_center_x = int((x + w) / 2)
            bottom_center_y = h

            # Draw a red dot at the bottom center
            cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)

            bot_mid_point = int(x + (w - x) / 2), int(h)
            
            for box_id in boxes_ids:
                x, y, w, h, id = box_id
                cv2.rectangle(frame, (x, y), (w, h), (255, 0, 255), 1)
                cv2.putText(frame, str(id), (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2)

                # Calculate the bottom center point
                bottom_center_x = int((x + w) / 2)
                bottom_center_y = h

                # Draw a red dot at the bottom center
                cv2.circle(frame, (bottom_center_x, bottom_center_y), 5, (0, 0, 255), -1)

                bot_mid_point = int(x + (w - x) / 2), int(h)
                
                # Calculate distances for each line
                distances_for_person = []
                for m, c in line_equations:
                    distance = distance_point_line(m, c, bot_mid_point[0], bot_mid_point[1])
                    distances_for_person.append(distance)
                
                # Update areas based on distances
                for idx, distance in enumerate(distances_for_person):
                    area_variable_name = f'area{idx+1}'
                    if distance < 10:  # Adjust the threshold as needed
                        areas[area_variable_name].add(id)                
                for line in areas:
                    print(line, len(areas[line]))
    
#      Showing the frame
        cv2.imshow('FRAME', frame)
    
        if cv2.waitKey(1) & 0xFF == 27:
           break

cap.release()
cv2.destroyAllWindows()
