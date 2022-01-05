import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


class DistanceCalculator:
    """calculate the distance (cm) between user's eyes and laptop screen using webcam 
    """

    # colors to use (in BGR)
    colors = [(76, 168, 240), (255, 0, 255), (255, 255, 0)]
    # instantiation face detection solution
    face_detection = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.75)

    @staticmethod
    def draw_bbox(img, bbox, color, l=30, t=5, rt=1):
        """draw bounding box around user(s) face

        Args:
            img (numpy ndarray): video frame
            bbox (tuple): bounding box data (x,y,width, height)
            color (tuple): color in BGR
            l (int, optional): corners lines length. Defaults to 30.
            t (int, optional): corners lines thickness. Defaults to 5.
            rt (int, optional): bounding box thickness. Defaults to 1.
        """
        # draw bbox
        x, y, w, h = bbox
        x1, y1 = x + w, y + h

        cv2.rectangle(img, bbox, color, rt)
        # top left
        cv2.line(img, (x, y), (x + l, y), color, t)
        cv2.line(img, (x, y), (x, y + l), color, t)
        # top right
        cv2.line(img, (x1, y), (x1 - l, y), color, t)
        cv2.line(img, (x1, y), (x1, y + l), color, t)
        # bottom left
        cv2.line(img, (x, y1), (x + l, y1), color, t)
        cv2.line(img, (x, y1), (x, y1 - l), color, t)
        # bottom right
        cv2.line(img, (x1, y1), (x1 - l, y1), color, t)
        cv2.line(img, (x1, y1), (x1, y1 - l), color, t)

    @staticmethod
    def draw_dist_between_eyes(img, center_left, center_right, color, distance_value):
        """draw a line between user's eyes and annotate the distance (pixel) between them

        Args:
            img (numpy ndarray): video frame
            center_left (tuple): left eye landmark (x,y)
            center_right (tuple): right eye landmark (x,y)
            color (tuple): color in BGR
            distance_value ([type]): distance between eyes (pixel)
        """
        # mark eyes
        cv2.circle(img, center_left, 1, color, thickness=8)
        cv2.circle(img, center_right, 1, color, thickness=8)

        # line between eyes
        cv2.line(img, center_left, center_right, color, 3)

        # add distance value
        cv2.putText(img, f'{int(distance_value)}',
                    (center_left[0], center_left[1] -
                     10), cv2.FONT_HERSHEY_PLAIN,
                    2, color, 2)

    def run_config(self):
        """it is used to for the initial configuration of the system where the user needs to measure few distances in cm corresponding to different distances in pixel  
        """

        # webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:
                    # get bbox data
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin*iw), int(bboxc.ymin *
                                                   ih), int(bboxc.width*iw), int(bboxc.height*ih)
                    bbox_list.append(bbox)

                    # get the eyes landmark
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x*iw), int(left_eye.y*ih)),
                                      (int(right_eye.x*iw), int(right_eye.y*ih))])

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):

                # calculate distance between left and right eye
                dist_between_eyes = np.sqrt(
                    (eye[0][1]-eye[1][1])**2 + (eye[0][0]-eye[1][0])**2)

                # draw bbox
                DistanceCalculator.draw_bbox(image, bbox, self.colors[0])

                # draw distace between eyes
                DistanceCalculator.draw_dist_between_eyes(
                    image, eye[0], eye[1], self.colors[0], dist_between_eyes)

            cv2.imshow('webcam', image)
            if cv2.waitKey(5) & 0xFF == ord('k'):
                break
        cap.release()

    def calculate_distance(self, distance_pixel, distance_cm):
        """calculate distance in cm between user's eyes and laptop screen

        Args:
            distance_pixel (list): distance between eyes in pixel
            distance_cm (list): distance to screen in cm
        """

        # get corrlation coffs
        coff = np.polyfit(distance_pixel, distance_cm, 2)

        # For webcam input:
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image)
            bbox_list, eyes_list = [], []
            if results.detections:
                for detection in results.detections:

                    # get bbox data
                    bboxc = detection.location_data.relative_bounding_box
                    ih, iw, ic = image.shape
                    bbox = int(bboxc.xmin*iw), int(bboxc.ymin *
                                                   ih), int(bboxc.width*iw), int(bboxc.height*ih)
                    bbox_list.append(bbox)

                    # get the eyes landmark
                    left_eye = detection.location_data.relative_keypoints[0]
                    right_eye = detection.location_data.relative_keypoints[1]
                    eyes_list.append([(int(left_eye.x*iw), int(left_eye.y*ih)),
                                      (int(right_eye.x*iw), int(right_eye.y*ih))])

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for bbox, eye in zip(bbox_list, eyes_list):

                # calculate distance between left and right eye
                dist_between_eyes = np.sqrt(
                    (eye[0][1]-eye[1][1])**2 + (eye[0][0]-eye[1][0])**2)

                # calculate distance in cm
                a, b, c = coff
                distance_cm = a*dist_between_eyes**2+b*dist_between_eyes+c

                if distance_cm > 51:  # for safe use, distance to screen should be grater than 51 cm
                    # draw bbox
                    DistanceCalculator.draw_bbox(image, bbox, self.colors[2])
                    # add distance in cm
                    cv2.putText(image, f'{int(distance_cm)} cm - safe',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                                2, self.colors[2], 2)

                else:
                    # draw bbox
                    DistanceCalculator.draw_bbox(image, bbox, self.colors[1])
                    # add distance in cm
                    cv2.putText(image, f'{int(distance_cm)} cm - too close',
                                (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_PLAIN,
                                2, self.colors[1], 2)

            cv2.imshow('webcam', image)
            if cv2.waitKey(5) & 0xFF == ord('k'):
                break
        cap.release()


if __name__ == '__main__':
    # # step-1: initial configuration
    # eye_screen_distance = DistanceCalculator()
    # eye_screen_distance.run_config()

    # step-2: get eye to screen distance
    distance_df = pd.read_csv('distance_xy.csv')
    eye_screen_distance = DistanceCalculator()
    eye_screen_distance.calculate_distance(
        distance_df['distance_pixel'], distance_df['distance_cm'])
