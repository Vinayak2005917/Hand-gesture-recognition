import cv2 as cv
import mediapipe as mp
import time
import numpy as np


#open the camera and record the guesture for 5 seconds

palm_landmark_data = []
cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils


start_time = time.time()
while True:
    elapsed_time = time.time() - start_time
    success, img = cap.read()
    img = cv.flip(img, 1)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            palm_landmark_data = ([[id, lm.x, lm.y] for id, lm in enumerate(handLms.landmark)])

    cv.putText(img, "Recording...", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv.putText(img, f"Time: {elapsed_time:.2f}s", (10, 70), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q') or elapsed_time > 5:
        break

cap.release()
cv.destroyAllWindows()


for i in range(len(palm_landmark_data)):
    for j in range(i + 1, len(palm_landmark_data)):
        dist = ((palm_landmark_data[i][1] - palm_landmark_data[j][1]) ** 2 + (palm_landmark_data[i][2] - palm_landmark_data[j][2]) ** 2) ** 0.5
        with open("Test.txt", "a") as f:
            f.write(f"{i},{j},{dist}\n")