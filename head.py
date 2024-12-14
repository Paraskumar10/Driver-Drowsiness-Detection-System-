import cv2
import dlib
import numpy as np
import pygame

# Initialize pygame for sound
pygame.mixer.init()
pygame.mixer.music.load("music.wav")  # Replace "alarm.mp3" with your alarm sound file

# Initialize face detector and facial landmarks predictor
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # You need to download this file

# Initialize constants
EYE_AR_THRESHOLD = 0.25  # Adjust this threshold as needed
HEAD_POSE_THRESHOLD = 30  # Adjust this threshold for head pose detection
FRAME_COUNTER = 20  # Number of consecutive frames for drowsy detection
ALARM_ON = False

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance(eye[1], eye[5])
    B = distance(eye[2], eye[4])
    C = distance(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate Euclidean distance between two points
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Function to check if the driver's head is tilted sideways
def is_head_tilted_sideways(landmarks):
    left_eye_x = landmarks.part(36).x
    right_eye_x = landmarks.part(45).x
    nose_x = landmarks.part(30).x

    face_width = distance(landmarks.part(0), landmarks.part(16))

    # Calculate the angle between the line connecting the eyes and the line connecting the eyes to the nose
    eye_to_nose_vector = np.array([nose_x - (left_eye_x + right_eye_x) / 2, 0])
    eye_vector = np.array([right_eye_x - left_eye_x, 0])

    angle = np.arccos(np.dot(eye_vector, eye_to_nose_vector) / (np.linalg.norm(eye_vector) * np.linalg.norm(eye_to_nose_vector)))
    angle_degrees = np.degrees(angle)

    return angle_degrees > HEAD_POSE_THRESHOLD

# Start capturing video from the webcam
cap = cv2.VideoCapture(0)

frame_counter = 0

while True:
    ret, frame = cap.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        # Detect facial landmarks
        landmarks = landmark_predictor(gray, face)

        # Check if the driver's head is tilted sideways
        if is_head_tilted_sideways(landmarks):
            frame_counter += 1

            if frame_counter >= FRAME_COUNTER:
                if not ALARM_ON:
                    pygame.mixer.music.play(-1)  # Play alarm sound continuously
                    ALARM_ON = True

                cv2.putText(frame, "DRIVER HEAD TILTED SIDEWAYS!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            frame_counter = 0
            ALARM_ON = False
            pygame.mixer.music.stop()

    cv2.imshow("Head Tilted Sideways Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()