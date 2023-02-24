import cv2, os
import numpy as np
from utils import validate_path, get_color_mask, detect_blob_from_mask, box_all_keypoints, draw_points, get_next_color

RESIZE_RATIO = 2
LOWER_RED = np.array([130, 120, 70])
UPPER_RED = np.array([180, 255, 255])

vid_path = validate_path('assets/juggle2.mp4')
output_path = os.path.join(os.path.dirname(vid_path), 'output_' + os.path.basename(vid_path)[:-4] + '.mov')

cap = cv2.VideoCapture(vid_path)
w, h = int(cap.get(3)), int(cap.get(4))


fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h // 2), True)

frame_right = np.zeros((h // 2, w // 2, 3), np.uint8)
color = (255, 0, 0)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = np.zeros((h // 2, w, 3), np.uint8)
    

    frame_left = cv2.resize(frame, (0,0), fx=1 / RESIZE_RATIO, fy=1 / RESIZE_RATIO)

    mask = get_color_mask(frame_left, LOWER_RED, UPPER_RED)
    keypoints = detect_blob_from_mask(mask)

    frame_left = box_all_keypoints(keypoints, frame_left)
    frame_right = draw_points(frame_right, keypoints, color)

    image[:, :w // RESIZE_RATIO] = frame_left
    image[:, w // RESIZE_RATIO:] = frame_right
    cv2.imshow('vid', image)
    out.write(image)

    color = get_next_color(color)
    if cv2.waitKey(1000 // fps) == ord('q'):
        break

print(image.shape)
cap.release()
out.release()
cv2.destroyAllWindows()