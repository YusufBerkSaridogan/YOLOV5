import cv2
import numpy as np

# Initialize variables
points = []

def draw_polygon(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        if len(points) > 1:
            cv2.line(image, points[-2], (x, y), (255, 255, 255), 2)
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)
        if len(points) == 4:
            cv2.line(image, points[2], points[-1], (255, 255, 255), 2)
            cv2.polylines(image, [np.array(points, np.int32)], isClosed=True, color=(255, 255, 255), thickness=2)
            points = []

# Create a black image
image = np.zeros((600, 800, 3), np.uint8)

cv2.namedWindow('Draw Polygons with Lines')
cv2.setMouseCallback('Draw Polygons with Lines', draw_polygon)

while True:
    cv2.imshow('Draw Polygons with Lines', image)

    key = cv2.waitKey(1)
    if key == 27:
        break

cv2.destroyAllWindows()
