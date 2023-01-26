import os
import setuptools

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    REQUIRED = f.read().split('\n')

setuptools.setup(
    name="gaze_orientation_demo",
    version="4",
    author="Paul Ang",
    long_description="A demo of gaze orientation estimation.",
    description="A demo of gaze orientation estimation.",
    packages=["gaze_orientation_demo"],
    install_requires=REQUIRED,
    python_requires='>=3.7.0'
)
