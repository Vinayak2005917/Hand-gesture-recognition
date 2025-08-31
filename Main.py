import cv2 as cv
import mediapipe as mp
import time
from utils import hand_bounding_box

cap = cv.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mpDraw = mp.solutions.drawing_utils

tolerance = 0.075   # distance tolerance
similarity_threshold = 0.75#% of distances that must match

while True:
    success, img = cap.read()
    img = cv.flip(img, 1)
    imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # Get landmarks (normalized x,y in range [0,1])
            landmarks = [(lm.x, lm.y) for lm in handLms.landmark]

            # Compute all pairwise distances
            hand_landmark_data = []
            for i in range(len(landmarks)):
                for j in range(i + 1, len(landmarks)):
                    dist = ((landmarks[i][0] - landmarks[j][0]) ** 2 +
                            (landmarks[i][1] - landmarks[j][1]) ** 2) ** 0.5
                    hand_landmark_data.append((i, j, dist))



                          #Palm detection
#-------------------------------------------------------------------------------------------

            # Load reference palm data
            with open("Gestures/palm_data.txt", "r") as f:
                palm_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(palm_data)

            for ref in palm_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold:
                cv.putText(img, "Palm Detected!!", 
                           (int(landmarks[12][0] * img.shape[1]) - 100,
                            int(landmarks[12][1] * img.shape[0]) - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
                # Add a rectangle around the palm,
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        hand_bounding_box(handLms, img)


                          #Peace sign detection
#-------------------------------------------------------------------------------------------
            # Load reference peace sign data
            with open("Gestures/peace_sign_data.txt", "r") as f:
                peace_sign_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(peace_sign_data)

            for ref in peace_sign_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold:
                cv.putText(img, "Peace sign Detected!!", 
                           (int(landmarks[12][0] * img.shape[1]) - 200,
                            int(landmarks[12][1] * img.shape[0]) - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
                # Add a rectangle around the palm
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        hand_bounding_box(handLms, img)



                          #Fist detection
#-------------------------------------------------------------------------------------------
            tolerance_fist = 0.05   # distance tolerance
            similarity_threshold_fist = 0.9#% of distances that must match
            # Load reference fist data
            with open("Gestures/fist_data.txt", "r") as f:
                fist_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(fist_data)

            for ref in fist_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance_fist:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold_fist:
                cv.putText(img, "Fist Detected!!", 
                           (int(landmarks[12][0] * img.shape[1]) - 100,
                            int(landmarks[12][1] * img.shape[0]) - 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
                # Add a rectangle around the palm
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        hand_bounding_box(handLms, img)



                          #Thumbs up detection
#-------------------------------------------------------------------------------------------
            tolerance_thumbs_up = 0.06   # distance tolerance
            similarity_threshold_thumbs_up = 0.9#% of distances that must match
            # Load reference thumbs up data
            with open("Gestures/thumbs_up.txt", "r") as f:
                thumbs_up_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(thumbs_up_data)

            for ref in thumbs_up_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance_thumbs_up:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold_thumbs_up:
                cv.putText(img, "Thumbs Up Detected!!", 
                           (int(landmarks[12][0] * img.shape[1]) - 100,
                            int(landmarks[12][1] * img.shape[0]) - 120),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
                # Add a rectangle around the palm
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        hand_bounding_box(handLms, img)



        # Un-comment the following lines to enable test symbol detection
        '''
                          #test detection
#-------------------------------------------------------------------------------------------
            tolerance_test = 0.06   # distance tolerance
            similarity_threshold_test = 0.75#% of distances that must match
            # Load reference test data
            with open("Test.txt", "r") as f:
                test_data = [tuple(map(float, line.strip().split(","))) for line in f]

            # Compare with tolerance
            match_count = 0
            total_count = len(test_data)

            for ref in test_data:
                ref_i, ref_j, ref_dist = int(ref[0]), int(ref[1]), ref[2]

                for live in hand_landmark_data:
                    live_i, live_j, live_dist = live
                    if live_i == ref_i and live_j == ref_j:
                        if abs(live_dist - ref_dist) <= tolerance_test:
                            match_count += 1
                        break

            # Palm detected if enough distances match
            if match_count / total_count > similarity_threshold_test:
                cv.putText(img, "Test Item Detected!!", 
                           (int(landmarks[12][0] * img.shape[1]) - 100,
                            int(landmarks[12][1] * img.shape[0]) - 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,)
                # Add a rectangle around the palm
                if results.multi_hand_landmarks:
                    for handLms in results.multi_hand_landmarks:
                        hand_bounding_box(handLms, img)
        '''



    cv.imshow("Image", img)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()