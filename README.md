# A demo for Gaze orientation estimation

This repository implements two kinds of gaze estimation: based on the head position and based on the irises position. 

## Installation using pip

`pip install gaze-orientation-demo --extra-index-url https://test.pypi.org/simple`

Note: A tagged release will automatically published to the test pypi via a GitHub action.

## Real-time gaze estimation based on web-cam video.
    from gaze_orientation_demo import GazeEstimation

    gaze_estimation = GazeEstimation()
    gaze_estimation.webcam_live_gaze_estimation()