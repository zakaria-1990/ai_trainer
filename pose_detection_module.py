import cv2
import mediapipe as mp
import math
import numpy as np


class PoseDetector:
    def __init__(self, mode=False, up_body=False, smooth=True, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.up_body = up_body
        self.smooth = smooth
        self.detection_con = detection_con
        self.track_con = track_con

        self.mp_draw = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.mode,
            enable_segmentation=self.up_body,
            smooth_landmarks=self.smooth,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.track_con,
        )

    def find_pose(self, img, draw=True):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(img)
        if self.results.pose_landmarks:
            if draw:
                self.mp_draw.draw_landmarks(img,
                                            self.results.pose_landmarks,
                                            self.mp_pose.POSE_CONNECTIONS,)
        return img

    def find_position(self, img, draw=True):
        self.landmark_list = []
        if self.results.pose_landmarks:
            for idx, landmark in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                self.landmark_list.append([idx, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
            return self.landmark_list

    def find_angle(self, img, p1, p2, p3, draw=True):
        x1, y1 = self.landmark_list[p1][1:]
        x2, y2 = self.landmark_list[p2][1:]
        x3, y3 = self.landmark_list[p3][1:]

        radians = np.arctan2(y3 - y2, x3 - x2) - np.arctan2(y1 - y2, x1 - x2)
        angle = np.abs(radians * 180 / np.pi)
        # angle = math.degrees(math.atan2(y3-y2, x3-x2) - math.atan2(y1-y2, x1-x2))

        if angle > 180.0:
            angle = 360.0 - angle

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
            cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), 3)
            cv2.circle(img, (x1, y1), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x1, y1), 15, (0, 0, 255), 2)
            cv2.circle(img, (x2, y2), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (0, 0, 255), 2)
            cv2.circle(img, (x3, y3), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (x3, y3), 15, (0, 0, 255), 2)
            cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        return angle


def main():
    cap = cv2.VideoCapture(0)
    pose_detector = PoseDetector()

    while cap.isOpened():
        success, img = cap.read()
        img = pose_detector.find_pose(img)
        landmark_list = pose_detector.find_position(img, draw=False)
        if landmark_list != None:
            print(landmark_list[14])
            cv2.circle(img, (landmark_list[14][1], landmark_list[14][2]), 15, (0, 0, 255), cv2.FILLED)

        cv2.imshow('pose detection', img)
        if cv2.waitKey(5) & 0XFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()