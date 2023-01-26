import cv2
import mediapipe as mp
import numpy as np

from gaze_orientation_demo.head_pose import HeadPose


class GazeEstimation():
    def get_face_landmarks(self, frame: np.array, face_mesh: mp.solutions.face_mesh.FaceMesh):
        ''' Returns the meshpipe's face landmarks. This function assumes one face only in a given frame.
        :param image: a frame (in BGR format) captured from the webcam
        :param face_mesh: a Meshpipe's face mesh object
        :return: face landmarks
        '''
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame)

        if results.multi_face_landmarks:
            return results.multi_face_landmarks[0]
        else:
            return None

    def webcam_live_gaze_estimation(self, video_device_id: int = 0, draw_meshes: bool = False):
        ''' Capture frames from the video device and performs gaze estimation in real-time.
        :param video_device_id: the id of video device. Set ID = 0 for the in-built webcam.
        :param draw_meshes: whether to overlay the meshes on the face or not.
        '''
        face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                                    min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5)
        head_pose = HeadPose()

        # Live capture from the camera
        cap = cv2.VideoCapture(video_device_id)
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)  # flip for selfie view

            # Obtain face landmarks
            landmarks = self.get_face_landmarks(frame, face_mesh)

            # Draw meshes
            if draw_meshes:
                # Draw face mesh
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_tesselation_style())

                # Draw face border, eyebrows, lips, and eyes
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_contours_style())

                # Draw irises
                mp.solutions.drawing_utils.draw_landmarks(
                    image=frame,
                    landmark_list=landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_styles
                    .get_default_face_mesh_iris_connections_style())

            if landmarks is not None:
                # Gaze estimation based on head's position
                frame = head_pose.estimate_direction(frame, landmarks)

                # TODO:  Pose estimation based on iris

            # Display frame on screen
            cv2.imshow('Live gaze estimation', frame)
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # fixes a bug where the window do not close properly on macOS.


if __name__ == "__main__":
    gaze_estimator = GazeEstimation()
    gaze_estimator.webcam_live_gaze_estimation(draw_meshes=False)
