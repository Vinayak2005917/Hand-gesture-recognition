import cv2 as cv
import mediapipe as mp
import time
import numpy as np

def hand_bounding_box(landmarks, img , color = (0, 255, 0)):
    h, w, _ = img.shape

    # Convert normalized landmark coords (0-1) → pixel coords
    xs = [int(lm.x * w) for lm in landmarks.landmark]
    ys = [int(lm.y * h) for lm in landmarks.landmark]

    # Find extremes
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # Draw rectangle
    cv.rectangle(img, (min_x, min_y), (max_x, max_y), color, 2)

    return (min_x, min_y, max_x, max_y)  # return box if needed