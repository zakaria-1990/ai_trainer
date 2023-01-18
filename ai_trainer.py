import cv2
import os
import mediapipe as mp
import numpy
import pose_detection_module as pm

BASE_DIR = '/home/zakaria/Downloads'
FILE = 'Crossfit - 66991.mp4'

path = os.path.join(BASE_DIR, FILE)

cap = cv2.VideoCapture(path)
pose_detector = pm.PoseDetector()

counter = 0
stage = 'DOWN'

while cap.isOpened():
    success, img = cap.read()

    img = pose_detector.find_pose(img, draw=False)
    landmark_list = pose_detector.find_position(img, draw=False)

    if landmark_list != None:
        # get angle between right wrist, right elbow and right shoulder
        angle = pose_detector.find_angle(img, 16, 14, 12, draw=False)

        if angle > 100.0 and stage == 'UP':
            stage = 'DOWN'
        if angle < 60.0 and stage == 'DOWN':
            counter += 1
            stage = 'UP'
        print(counter)
        print(stage)

        cv2.putText(img, str(counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('ai_trainer', img)
    if cv2.waitKey(5) & 0XFF == 27:
        break
cap.release()
cv2.destroyAllWindows()