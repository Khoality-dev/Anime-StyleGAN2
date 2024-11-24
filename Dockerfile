FROM ubuntu:22.04

RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    software-properties-common

RUN add-apt-repository ppa:deadsnakes/ppa && apt-get update && apt-get install -y python3.10 \
    python3.10-dev \
    python3.10-venv

RUN mkdir /project
WORKDIR /project

COPY . /project/

RUN python3.10 -m venv venv
RUN . venv/bin/activate && pip install --upgrade pip 
RUN venv/bin/pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118 
RUN venv/bin/pip install -r requirements.txt

CMD ["venv/bin/python", "train.py", "-n", "-b 8", "-l 1", "-p 5000", "-c 5000"]