import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList
import cv2


# This class estimates the gaze direction based on the irises' position.
class IrisPose:
    def __init__(self):
        pass

    def estimate_direction(self, frame: np.array, landmarks: NormalizedLandmarkList, draw_pose_direction: bool = True,
                           distance_threshold: int = 3):
        '''
            :param frame: a frame from the webcam.
            :param landmarks: the NormalizedLandmarkList from Meshpipe.
            :param draw_pose_direction: whether to draw a line from the nose to indicate the pose direction.
            :param distance_threshold: the x and y distance threshold for determining the gaze direction.
        '''

        # Iris positions
        left_iris_image_coor = self.get_average_coors(landmarks, (474, 475, 476, 477), frame.shape)
        right_iris_image_coor = self.get_average_coors(landmarks, (468, 469, 470, 471, 472), frame.shape)

        # Center point of both irises
        between_iris_image_coor = ((left_iris_image_coor[0] + right_iris_image_coor[0])//2, (left_iris_image_coor[1] + right_iris_image_coor[1])//2)

        # Eye center positions
        left_eye_center_coor = self.get_average_coors(landmarks, (
        263, 249, 390, 373, 374, 380, 381, 382, 362, 263, 466, 388, 387, 386, 385, 384, 398), frame.shape)
        right_eye_center_coor = self.get_average_coors(landmarks, (
        33, 7, 163, 144, 145, 153, 154, 155, 133, 33, 246, 161, 160, 159, 158, 157, 173, 133), frame.shape)

        # Center point of both eyes' center
        between_eyes_image_coor = ((left_eye_center_coor[0] + right_eye_center_coor[0]) // 2,
                                   (left_eye_center_coor[1] + right_eye_center_coor[1]) // 2)

        distance_between_iris_eyes_center = np.array(between_eyes_image_coor) - np.array(between_iris_image_coor)
        x_distance = distance_between_iris_eyes_center[0]
        y_distance = distance_between_iris_eyes_center[1]

        # Get direction
        gaze_direction = ""
        if y_distance > distance_threshold:
            gaze_direction = "Top"
        elif y_distance < -distance_threshold:
            gaze_direction = "Bottom"
        elif x_distance > distance_threshold:
            gaze_direction = "Left"
        elif x_distance < -distance_threshold:
            gaze_direction = "Right"
        else:
            gaze_direction = "Center"

        # Annotate
        frame = cv2.line(frame, left_iris_image_coor, left_iris_image_coor, (0, 0, 255), 3)
        frame = cv2.line(frame, right_iris_image_coor, right_iris_image_coor, (0, 0, 255), 3)
        frame = cv2.line(frame, between_eyes_image_coor, between_eyes_image_coor, (255, 0, 0), 3)
        frame = cv2.line(frame, between_iris_image_coor, between_iris_image_coor, (0, 0, 255), 3)
        frame = cv2.putText(frame, f"X distance center of eyes and irises: {x_distance}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)
        frame = cv2.putText(frame, f"Y distance center of eyes and irises: {y_distance}", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)
        frame = cv2.putText(frame, f"Iris gaze direction: {gaze_direction}", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, 2)

        return frame



    def get_average_coors(self, landmarks, landmark_idxs, shape):
        '''
        Calculate the average of multiple image coordinates.
        :param landmarks: the meshpipe's face landmarks.
        :param landmark_idxs: the indices.
        :param shape: the frame dimensions
        :return:
        '''
        total_x = []
        total_y = []
        for idx in landmark_idxs:
            coor = self.denorm_coor(landmarks.landmark[idx], shape)
            total_x.append(coor[0])
            total_y.append(coor[1])

        avg_x = int(np.mean(total_x))
        avg_y = int(np.mean(total_y))
        return (avg_x, avg_y)

    def denorm_coor(self, coor, shape):
        '''
        Return the normalized coordinates to the original scale.
        :param coor: x and y coordinates
        :param shape: the frame's dimensions.
        :return: the de-normalized xy coordinates
        '''
        return (int(coor.x * shape[1]), int(coor.y * shape[0]))
