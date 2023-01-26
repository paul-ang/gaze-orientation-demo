import cv2
import numpy as np
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmarkList


# This class estimates the gaze direction based on the head's position.
class HeadPose:
    def __init__(self):
        pass

    def estimate_direction(self, frame: np.array, landmarks: NormalizedLandmarkList, draw_pose_direction: bool = True,
                           angle_threshold: int = 25):
        '''
        :param frame: a frame from the webcam.
        :param landmarks: the NormalizedLandmarkList from Meshpipe.
        :param draw_pose_direction: whether to draw a line from the nose to indicate the pose direction.
        :param angle_threshold: the angle threshold of x and y rotations to determine the direction.
        :return: the annotated frame. The pose direction is annotated in the frame.
        '''

        # Get transformation vectors from world to image
        rot_vector, trans_vector, camera_matrix, dist_coeffs = self.get_world_to_image_trans(frame, landmarks)

        # Draw nose
        if draw_pose_direction:
            # Transform nose end's world point to image point
            nose_end_coor, _ = cv2.projectPoints(np.array([(0.0, 0.0, 500.0)]), rot_vector, trans_vector,
                                                 camera_matrix, dist_coeffs)
            start_nose_coor = self.denorm_coor(landmarks.landmark[4], frame.shape)
            p1 = (int(start_nose_coor[0]), int(start_nose_coor[1]))
            p2 = (int(nose_end_coor[0][0][0]), int(nose_end_coor[0][0][1]))
            frame = cv2.line(frame, p1, p2, (255, 0, 0), 2)

        # Get rotation angles
        x_angle, y_angle, z_angle = self.xyz_rot_angles(rot_vector)

        head_direction = ""
        if y_angle > angle_threshold:
            head_direction = "Left"
        elif y_angle < -angle_threshold:
            head_direction = "Right"
        elif x_angle > angle_threshold:
            head_direction = "Bottom"
        elif x_angle < -angle_threshold:
            head_direction = "Top"
        else:
            head_direction = "Center"

        # Put text regarding pose
        frame = cv2.putText(frame, f"Head x rotation: {x_angle:.0f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, 2)
        frame = cv2.putText(frame, f"Head y rotation: {y_angle:.0f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, 2)
        frame = cv2.putText(frame, f"Head z rotation: {z_angle:.0f}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 0), 1, 2)
        frame = cv2.putText(frame, f"Head pose: {head_direction}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                            1, 2)

        return frame

    def xyz_rot_angles(self, rot_vector: np.array):
        '''
        Converts the rotation vector to rotation matrix, and then obtain the angles.
        :param rot_vector: the rotation vector.
        :return: the rotation angles in the x,y,z directions.
        '''
        rot_mat, _ = cv2.Rodrigues(rot_vector)
        angles, _, _, _, _, _ = cv2.RQDecomp3x3(rot_mat)
        x_angle = angles[0]
        y_angle = angles[1]
        z_angle = angles[2]
        return x_angle, y_angle, z_angle

    def denorm_coor(self, coor, shape):
        '''
        Return the normalized coordinates to the original scale.
        :param coor: x and y coordinates
        :param shape: the frame's dimensions.
        :return: the de-normalized xy coordinates
        '''
        return (int(coor.x * shape[1]), int(coor.y * shape[0]))

    def get_world_to_image_trans(self, frame: np.array, landmarks: NormalizedLandmarkList):
        '''
        Reference: https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/
        :param frame: a frame from the webcam
        :param landmarks: face landmarks from Meshpipe.
        :return: rotation vector, translation vector, camera intrinsic parameters, distortion coefficients
        '''

        # 2D image coordinates of several facial landmarks
        image_points = np.array([
            self.denorm_coor(landmarks.landmark[4], frame.shape),  # Nose tip
            self.denorm_coor(landmarks.landmark[152], frame.shape),  # Chin
            self.denorm_coor(landmarks.landmark[263], frame.shape),  # Left eye left corner
            self.denorm_coor(landmarks.landmark[33], frame.shape),  # Right eye right corner
            self.denorm_coor(landmarks.landmark[287], frame.shape),  # Left Mouth corner
            self.denorm_coor(landmarks.landmark[57], frame.shape)  # Right mouth corner
        ], dtype="double")

        # 3D World coordinates based on a generic face model
        world_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-225.0, 170.0, -135.0),  # Left eye left corner
            (225.0, 170.0, -135.0),  # Right eye right corner
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])

        # Estimated camera attributes
        focal_length = frame.shape[1]
        center = (frame.shape[1] / 2, frame.shape[0] / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )

        dist_coeffs = np.zeros((4, 1))  # assuming no lens distortion

        # Get rotation and traslation vectors
        (success, rot_vector, trans_vector) = cv2.solvePnP(world_points, image_points, camera_matrix,
                                                           dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

        return rot_vector, trans_vector, camera_matrix, dist_coeffs
