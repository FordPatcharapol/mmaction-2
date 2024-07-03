import cv2
import numpy as np

# Initialize list to store coordinates of the polygons
points = []
polygon_lst = []
display_size = (640, 480)


def mouse_click(event, x, y, flags, param):
    """ Mouse callback function to capture points """
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
        cv2.imshow('Frame', frame)

        if len(points) > 1:
            cv2.line(frame, points[-2], points[-1], (255, 0, 0), 2)
            cv2.imshow('Frame', frame)


def save_polygons_to_file(polygon_list, filename="polygons.txt"):
    """ Save the list of polygons to a file """
    with open(filename, 'w') as f:
        for polygon in polygon_list:
            polygon_str = ', '.join(f"({x}, {y})" for x, y in polygon)
            f.write(f"{polygon_str}\n")


cap = cv2.VideoCapture('./videos/2024-05-14_12-07-40_1.mp4')
ret, frame = cap.read()
frame = cv2.resize(frame, display_size)

if not ret:
    print("Failed to grab frame")
    cap.release()
    exit()

cv2.namedWindow('Frame')
cv2.setMouseCallback('Frame', mouse_click)
cv2.imshow('Frame', frame)

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord('d'):
        if points:
            polygon_lst.append(points.copy())
            points.clear()
            frame_copy = frame.copy()
            cv2.imshow('Frame', frame_copy)  # Refresh frame
    elif key == ord('q'):
        save_polygons_to_file(polygon_lst)
        break
    elif key == 27:  # ESC key to exit without saving
        break

cap.release()
cv2.destroyAllWindows()
