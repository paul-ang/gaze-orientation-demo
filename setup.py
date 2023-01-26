import os
import setuptools

setuptools.setup(
    name="gaze_orientation_demo",
    version="4",
    author="Paul Ang",
    long_description="A demo of gaze orientation estimation.",
    description="A demo of gaze orientation estimation.",
    packages=["gaze_orientation_demo"],
    install_requires=['numpy==1.21.6', 'mediapipe==0.9.0.1', 'opencv-python==4.7.0.68'],
    python_requires='>=3.7.0'
)
