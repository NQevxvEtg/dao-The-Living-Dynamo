# dao/Dockerfile
# Use a CUDA-enabled base image from NVIDIA.
# Choose a version that matches your CUDA toolkit and GPU driver.
# https://hub.docker.com/r/nvidia/cuda/tags?name=12.8.1-cudnn-runtime-u
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set the DEBIAN_FRONTEND environment variable to noninteractive.
# This is the crucial step to prevent interactive prompts during apt-get installation.
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install prerequisites, add the deadsnakes PPA, and install Python 3.11.
# This command will now run without any interactive prompts.
RUN apt-get update && \
    apt-get install -y software-properties-common wget && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.11 python3.11-venv && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.11 get-pip.py && \
    rm get-pip.py && \
    rm -rf /var/lib/apt/lists/*

# Update the system's default 'python3' to point to the new 'python3.11'
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# First install PyTorch with CUDA support
RUN python3 -m pip install --no-cache-dir torch torchvision torchaudio

# Install pip-tools for pip-compile
RUN python3 -m pip install --no-cache-dir pip-tools

# Copy requirements.in and compile it
COPY requirements.in .
RUN pip-compile --output-file requirements.txt requirements.in

# Install compiled requirements
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# COPY ./src /app/src
# COPY ./models /app/models
EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]