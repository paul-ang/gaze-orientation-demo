# A demo for Gaze orientation estimation

This repository implements two kinds of gaze estimation: based on the head position and based on the irises position. 

# Usage instructions
## 1. Python package
- Ensure `Python >= 3.7`
- Note: A tagged release will automatically published to the test pypi via a GitHub action.

Run `pip install gaze-orientation-demo --extra-index-url https://test.pypi.org/simple`

**Example code:**

    from gaze_orientation_demo import GazeEstimation

    gaze_estimation = GazeEstimation()
    gaze_estimation.webcam_live_gaze_estimation()

## 2. Web server with Docker


##