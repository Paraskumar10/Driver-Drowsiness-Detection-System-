# Import necessary libraries
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2
import time

# Initialize the audio mixer
mixer.init()

# Load audio files
sleep_audio = mixer.Sound("sleep.mp3")
yawn_audio = mixer.Sound("yawn.mp3")
skull_audio = mixer.Sound("alarm.wav")

last_played_time = {
    "sleep_audio": 0,
    "yawn_audio": 0,
    "skull_audio": 0
}

# Define function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define function to calculate mouth aspect ratio (MAR)
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Define function to calculate skull aspect ratio (SAR)
def skull_aspect_ratio(eye, mouth):
    d_eye = distance.euclidean(eye[0], eye[3])
    d_mouth = distance.euclidean(mouth[14], mouth[18])
    sar = d_eye / d_mouth
    return sar

# Set threshold and frame check for drowsiness detection
thresh = 0.25
frame_check = 20

# Initialize counters for yawns and nods
yawn_count = 0
nod_count = 0

# Initialize face detection and shape prediction
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Define the indices for left eye, right eye, and mouth landmarks
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["mouth"]

# Initialize the webcam capture
cap = cv2.VideoCapture(0)

# Initialize a flag to track head movement detection
flag = 0

# Initialize the timestamp for the last subject detected
last_subject_detected_time = time.time()

# Main loop for video capture and processing
while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    
    # Check if a frame was successfully captured
    if not ret:
        print("Error: No frame received from the camera!")
        break

    # Resize the frame for faster processing
    frame = imutils.resize(frame, width=450)
    
    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    subjects = detect(gray, 0)

    # If no subjects are detected, check for head movement
    if len(subjects) == 0:
        if time.time() - last_subject_detected_time >= 1.0:
            cv2.putText(frame, "****Head Movement!****", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Frame", frame)
            skull_audio.play()
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

    # Process each detected subject
    for subject in subjects:
        # Predict facial landmarks for the subject
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        mouth = shape[mStart:mEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
        last_subject_detected_time = time.time()
        last_yawn_detected_time = time.time()
        
        
        # Display EAR on the frame
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Drowsiness detection
        if ear < thresh:
            flag += 1
            if flag >= frame_check:
                current_time = time.time()
                cv2.putText(frame, "****Sleeping!****", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                sleep_audio.play()
                # last_played_time["sleep_audio"] = current_time
        else:
            flag = 0
        
        # Yawning detection
        mar = mouth_aspect_ratio(mouth)
        if mar > 0.6:
            yawn_count += 1
            if yawn_count > 5:
                current_time = time.time()
                time.sleep(0.5)
                cv2.putText(frame, "****Yawning!****", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                yawn_audio.play()
                last_played_time["yawn_audio"] = current_time
        else:
            yawn_count = 0
        
        # Skull aspect ratio detection
        sar = skull_aspect_ratio(leftEye, mouth)
        if sar < 0.25:  # Adjust the threshold as needed
            cv2.putText(frame, "Skull Aspect Ratio Low", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
    # Display the processed frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release the webcam and close all OpenCV windows
cv2.destroyAllWindows()
cap.release()