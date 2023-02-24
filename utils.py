import os, cv2
import numpy as np

def validate_path(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError()
    return path

def get_color_mask(frame: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    erode_kernel = np.ones((3, 3),np.uint8)
    mask = cv2.erode(mask, erode_kernel, iterations=1)

    dilate_kernel = np.ones((15, 15),np.uint8)
    mask = cv2.dilate(mask, dilate_kernel, iterations=1)
    # return cv2.bitwise_and(frame, frame, mask=mask)
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

def detect_blob_from_mask(mask: np.ndarray) -> list[cv2.KeyPoint]:
    detector = _get_blob_detector()

    return detector.detect(mask)

def box_all_keypoints(keypoints: list[cv2.KeyPoint], output_img: np.ndarray):
    for keypoint in keypoints:
        output_img = _draw_rect_around_keypoint(output_img, keypoint)
    return output_img

def _get_blob_detector():
    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = False
    params.minThreshold = 65
    params.maxThreshold = 93
    params.blobColor = 0
    params.minArea = 10
    params.maxArea = 5000
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.minCircularity =.1
    params.maxCircularity = 1
    detector = cv2.SimpleBlobDetector_create(params)
    return detector 

def _draw_rect_around_keypoint(img:np.ndarray, keypoint: cv2.KeyPoint) -> np.ndarray:
    x, y = keypoint.pt
    color = (255, 50 ,0) # Blue boiiii
    thiccness = 1
    return cv2.rectangle(img, (int(x - 15), int(y - 15)), (int(x + 15), int(y + 15)), color, thiccness)

def _draw_dot_on_keypoint(img:np.ndarray, keypoint: cv2.KeyPoint, color: np.ndarray) -> np.ndarray:
    x, y = keypoint.pt

    return cv2.circle(img, (int(x), int(y)), 5, color, -1)

def draw_points(img:np.ndarray, keypoints: list[cv2.KeyPoint], color: np.ndarray) -> np.ndarray:
    for keypoint in keypoints:
        img = _draw_dot_on_keypoint(img, keypoint, color)
    return img

def get_next_color(prev_color: tuple):
    r, g, b = prev_color
    if (r and g) or (r and not g and not b):
        r -= 5
        g += 5
        return (r, g, b)
    if (g and b) or (g and not b and not r):
        g -= 5
        b += 5
        return (r, g, b)
    if (b and r) or (b and not r and not g):
        b -= 5
        r += 5
        return (r, g, b)