import sys
sys.path.append('../')

import base64
import numpy as np
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import mediapipe as mp
import cv2
from gaze_orientation_demo.head_pose import HeadPose
from gaze_orientation_demo.iris_pose import IrisPose

# Global variables
app = Flask(__name__)
socketio = SocketIO(app)
face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True,
                                            min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)
head_pose = HeadPose()
iris_pose = IrisPose()


def get_face_landmarks(frame: np.array, face_mesh: mp.solutions.face_mesh.FaceMesh):
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


@app.route('/')
def index():
    server_ip = request.headers.get('Host')
    return render_template('real-time.html', server_ip=server_ip)


@socketio.on('image')
def image(data_image):
    # Decode base64 string as image
    jpg_as_np = np.frombuffer(base64.b64decode(data_image), dtype=np.uint8)
    frame = cv2.imdecode(jpg_as_np, flags=1)
    frame = cv2.flip(frame, 1)  # flip for selfie view

    # Perform gaze estimation
    landmarks = get_face_landmarks(frame, face_mesh)
    if landmarks is not None:
        # Gaze estimation based on head's position
        frame = head_pose.estimate_direction(frame, landmarks)

        # Gaze estimation based on irises
        frame = iris_pose.estimate_direction(frame, landmarks)

    # Encode img back to base64
    frame_base64 = base64.b64encode(cv2.imencode('.jpg', frame)[1])

    # emit the frame back
    emit('response_back', str(frame_base64))


if __name__ == "__main__":
    socketio.run(app, host='0.0.0.0', port=2204, debug=True, ssl_context='adhoc', allow_unsafe_werkzeug=True)
