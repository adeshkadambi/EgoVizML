# Use the official Python image as the base image
FROM python:3.11-slim-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libopencv-dev \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch
RUN pip3 install torch torchvision torchaudio

# Install Detectron2
RUN pip3 install cython
RUN pip3 install 'git+https://github.com/facebookresearch/detectron2.git'

# Install OpenCV
RUN pip3 install opencv-python

# Set up the working directory
WORKDIR /workspace

# Create a non-root user to run applications within the container
RUN groupadd -g 1000 vscode && \
    useradd -u 1000 -g vscode -G sudo -m -s /bin/bash vscode && \
    echo "vscode ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/vscode

# Switch to the non-root user
USER vscode

# Expose the Jupyter notebook port
EXPOSE 8888

# Start a shell by default
CMD ["/bin/bash"]
