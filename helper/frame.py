import cv2

skeleton_structure = [
    (0, 1), (1, 3), (2, 4), (0, 2),  # Head
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 6), (5, 11), (6, 12), (11, 12),  # Body
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
]


def bbox_overlay_frame(frame, bboxes):
    for bbox in bboxes:
        color = (0, 255, 0)
        top_left = (int(bbox[0]), int(bbox[1]))
        bottom_right = (int(bbox[2]), int(bbox[3]))
        cv2.rectangle(frame, top_left, bottom_right, color, 2)
    return frame


def draw_skeleton(frame, detections, color=(0, 255, 0), thickness=2):
    for keypoints in detections:
        for start, end in skeleton_structure:
            if len(keypoints) > start and len(keypoints) > end and all(keypoints[start]) and all(keypoints[end]):
                pt1 = (int(keypoints[start][0]), int(keypoints[start][1]))
                pt2 = (int(keypoints[end][0]), int(keypoints[end][1]))
                cv2.line(frame, pt1, pt2, color, thickness)
    return frame
