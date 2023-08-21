import cv2
import torch
import numpy as np
import math
from tracker import Tracker
from deep_sort_realtime.deepsort_tracker import DeepSort

model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
tracker2 = DeepSort(max_age=30)

video_path = '638229488.mp4'
cap = cv2.VideoCapture(video_path)

# defining a area
area_1 = np.array([[450, 438], [363, 302], [590, 252], [692, 348]])
line_1 = np.array([[354, 363], [668, 287]])

areas = {}
crossed_ids = {}

# Initialize in-count and out-count to 0
total_count = 0

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can choose other codecs if needed
output_path = 'output_video.mp4'
output_size = (1020, 500)
out = cv2.VideoWriter(output_path, fourcc, 30.0, output_size)

# Global variables for polygon drawing
drawing_polygon = False
points = []
polygon_points = []
polygon_count = 0
# Global variables to keep track of mouse events and lines
drawing = False
start_point = (-1, -1)
lines = []
lines_coordinates = []  # List to store the coordinates of the lines

# Functions Section 
def draw_polygon(event, x, y, flags, param):
    global drawing_polygon, points, polygon_count
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_polygon = True
        points.append((x, y))
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        if len(points) > 1:
            cv2.line(image, points[-2], (x, y), (255, 255, 255), 2)
        if len(points) == 4:            
            cv2.line(image, points[2], points[-1], (255, 255, 255), 2)
            cv2.polylines(image, [np.array(points, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
            drawing_polygon = True                       
            polygon_count += 1
            polygon_points.append(points)
            points = []
            
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

#Activate the mouse event
cv2.namedWindow('Image')
cv2.setMouseCallback('Image', draw_polygon)


while True:
    if show_image:      
        # Drawing the lines
        for line in lines:
            cv2.line(image, line[0], line[1], (255, 255, 255), 2)

        # Draw coordinates above the lines
        for idx, line_coord in enumerate(lines_coordinates):
            text = f"{line_coord[0]} {line_coord[1]}"  # Example: "X1 Y1 X2 Y2"
            cv2.putText(image, text, line_coord[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            
        cv2.imshow('Image', image)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            show_image = False
            cv2.destroyWindow('Image')
            break

area_count = len(polygon_points)
areas = create_area_variables(area_count)




previous_tracked_objects = {}
tracker = Tracker()

while True:
    
    if True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
    
    # Draw the polygons on the frame
        for polygon in polygon_points:
            cv2.polylines(frame, [np.array(polygon, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
            
    # Drawing the lines
        for line in lines:
            cv2.line(frame, line[0], line[1], (255, 255, 255), 2)

    # Draw coordinates above the lines
        for idx, line_coord in enumerate(lines_coordinates):
            text = f"{line_coord[0]} {line_coord[1]}"  # Example: "X1 Y1 X2 Y2"
            cv2.putText(frame, text, line_coord[0], cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
            
            
        results = model(frame)         
        detection_list = []
        for index, row in results.pandas().xyxy[0].iterrows():
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            w = int(row['xmax']) - x1
            h = int(row['ymax']) - y1
            confidence = row['confidence']
            label = row['name']
            if 'person' in label:
                detection = ([x1, y1, w, h], confidence, label)
                detection_list.append(detection)        

        tracks = tracker2.update_tracks(detection_list, frame=frame)
        tracked_ids = []  # Renamed from "list"
        for track in tracks:
            bbox = track.to_tlbr()
            id = track.track_id
            tracked_ids.append(id)  # Append the track ID to the list
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
            cv2.putText(frame, str(id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200, (0, 255, 0), 1)
            bot_mid_point = int((bbox[0] + bbox[2]) / 2), int(bbox[3])  # Calculate bot_mid_point for the current track

            # Iterate through the polygon points and check if the bot_mid_point is within the area
            for idx, polygon in enumerate(polygon_points):
                area_variable_name = f'area{idx+1}'
                if cv2.pointPolygonTest(np.array(polygon), (bot_mid_point), False) >= 0 and id not in crossed_ids:
                    crossed_ids[id] = True
                    areas[area_variable_name].add(id)
                    total_count += 1          
                      
                      
                                
        cv2.putText(frame, f"Count: {total_count}", (10, 30), cv2.QT_FONT_NORMAL, 0.9, (255, 255, 255), 1)
        out.write(frame)
        
        
#      Showing the frame
        cv2.imshow('FRAME', frame)
    
        if cv2.waitKey(1) & 0xFF == 27:
           break
out.release()
cap.release()
cv2.destroyAllWindows()
