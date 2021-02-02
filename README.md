# Growing Virtual Creatures

[![Build Status](https://travis-ci.org/cfusting/conditional-growth.svg?branch=main)](https://travis-ci.org/cfusting/conditional-growth)

### Theory

![theory1](./docs/theory1.jpg)
![theory2](./docs/theory2.jpg)

### About

This package provides the necessary tooling to grow virtual creatures made from three-dimensional blocks called voxels (a 3d pixel). Starting with one or more voxels new voxels are iteratively added based on nearby voxels and the current position. Some voxels are energetic and pulse, causing the virtual creatures to move around.

### Running with Docker

#### Requirements
This project uses [voxcraft-sim](https://github.com/voxcraft/voxcraft-sim) to simulate voxels. When building voxcraft-sim the makefile checks if a GPU is available. Thus it is necessary for docker build to be able to see your GPU. To that end install and configure the [nvidia-container-runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

#### Installing the nvidia runtime

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

#### Build

Clone this repository and navigate into the root folder.

```bash
docker build -t con-grow .
```

#### Run

Run the optimization script.

```bash
docker run --rm --gpus all con-grow python experiments/grow_up/rl_optimize.py
```

