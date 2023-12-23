FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

WORKDIR /workspace

# install python 3.8 and OpenCV dependencies
RUN apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3.8-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Add the library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu/

# install python packages
RUN python3.8 -m pip install --upgrade pip

# install pytorch
RUN pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113

# install other packages
COPY requirements.txt .

RUN pip3 install -r requirements.txt

# copy source code
COPY . .

# build and develop
RUN cd lib && python3.8 setup.py build develop