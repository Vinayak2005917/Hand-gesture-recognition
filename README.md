# Vinayak Mishra Bhatiyani Astute Intelligence Internship Assesment 

# Hand Gesture Recognition

This project uses computer vision and machine learning to recognize hand gestures from webcam images. It leverages OpenCV and MediaPipe for hand tracking and gesture detection.


## Video Demonstration

https://youtu.be/uYVTof76p_Q

## Technology Justification
For hand detection, I have chosen MediaPipe Hands with OpenCV. MediaPipe provides highly optimized real-time hand tracking with 21 landmarks per hand, making it lightweight and faster than custom deep learning models. OpenCV was used for image processing, drawing landmarks, and video capture. Together, they allowed efficient implementation without needing large datasets or heavy GPU computation, making them ideal for real-time gesture recognition.

## Gesture Logic Explanation
The system identifies four gestures using the positions of the 21 hand landmarks. By analyzing relative distances between all the points and comparing with the data in the gestures folder (tolerance and similarity threshold were applied and can be modified).


## Project Structure
- `Main.py`: Main notebook/script for running gesture recognition.
- `New_gesture.py`: Notebook/script for adding or testing new gestures.
- `utils.py`: Contains utility functions, such as drawing bounding boxes around detected hands using MediaPipe landmarks and OpenCV.
- `Gestures/`: Contains data files for different hand gestures (e.g., fist, palm, peace sign, thumbs up).
- `requirements.txt`: Lists required Python packages (OpenCV, MediaPipe).

## Getting Started
1. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
2. Run `Main.py` to start gesture recognition.
3. Use `New_gesture.py` to add or test new gestures. This file will open a capture from you primary camera for 5 secounds and records and saves the hand data to a test file, then you have to un-comment a section in the main.py.

## Requirements
- Python 3.10 (Since media pipe dosen't yet support a higher python version)
- OpenCV
- MediaPipe

---
Feel free to extend the project by adding new gesture data files in the `Gestures/` folder and updating the recognition logic.
