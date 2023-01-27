# A demo for Gaze orientation estimation

This repository implements two kinds of gaze estimation: based on the head position and based on the irises position. 

# Usage instructions
## 1. Python package

Ensure `Python >= 3.7`. Note: A tagged release will automatically published to the test pypi via a GitHub action.

Install using `pip install gaze-orientation-demo --extra-index-url https://test.pypi.org/simple`

**Example code:**

    from gaze_orientation_demo import GazeEstimation

    gaze_estimation = GazeEstimation()
    gaze_estimation.webcam_live_gaze_estimation()

## 2. Web server with Docker

Tested on _desktop_ Chrome browser and _iPhone_ Safari browser.

1. Build the Dockerfile: `docker build -t gaze-demo .`
2. Create a container: `docker run -p 2204:2204 gaze-demo`
3. Access the website on the host machine via localhost (http://127.0.0.1:2204) or on other computers via the host's private IP (e.g., http://192.168.8.199:2204).
