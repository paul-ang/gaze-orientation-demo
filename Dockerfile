FROM python:3.7-slim

ENV PYTHONFAULTHANDLER=1 \
    PYTHONHASHSEED=random \
    PYTHONUNBUFFERED=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

COPY . .

RUN apt-get update && apt-get install libgl1-mesa-glx ffmpeg libsm6 libxext6  -y

RUN pip install -r requirements.txt

WORKDIR examples/

CMD ["python", "flask_server.py"]