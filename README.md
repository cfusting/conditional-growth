# Growing Virtual Creatures

[![Build Status](https://travis-ci.com/cfusting/conditional-growth.svg?branch=main)](https://travis-ci.com/cfusting/conditional-growth)

### About

This package provides the necessary tooling to grow virtual creatures made from three-dimensional blocks called voxels (3d pixels). Starting with one or more voxels new voxels are iteratively added based on the composition of nearby voxels and the current position. In the simulation environment some voxels are energetic and pulse, causing the virtual creatures to move around.

### Theory

![theory1](./docs/theory1.jpg)
![theory2](./docs/theory2.jpg)

### Building with Docker

#### Requirements
This project uses [voxcraft-sim](https://github.com/voxcraft/voxcraft-sim) to simulate voxels. When building voxcraft-sim the makefile checks if a GPU is available. Thus it is necessary for docker build to be able to see your GPU. To that end install and configure the [nvidia-container-runtime](https://stackoverflow.com/questions/59691207/docker-build-with-nvidia-runtime).

#### Installing the Nvidia runtime

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
docker build -t grow .
```

### Optimizing a Creature

In this example we will build a creature for which surface area is maximized and volume is minimized.

#### Run

Run the optimization script, storing the results in the host environment's /tmp directory.

```bash
docker run --rm --gpus all -v /tmp:/tmp --shm-size 2G grow python scripts/grow/optimize_grid.py
```

#### Metrics

Metrics are captured by the [ray](https://docs.ray.io/en/master/) framework in /tmp/ray_results/expname where expname is specified in the optimize_grid.py script. The easiest way to view the metrics is to use tensorboard. For example:

```bash
docker run -p 6006:6006 --rm tensorflow/tensorflow tensorboard --logdir /tmp/ray_results/badger
```

![tensorboard](./docs/tensorboard.png)


#### Creatures

Uncommenting `monitor=True` in optimize_grid.py will enable the recording of a creature being built. The resulting movies can be found in /tmp/ray_results/expname/trialname (as can all the other logs). Refer to [RLlib](https://docs.ray.io/en/master/rllib.html) for more details. 

Due to a memory leak in vtk (which is the graphics library used to create the movies), enabling monitoring will eventually cause the trial to crash. To avoid this run your experiment until convergence and turn on monitoring after loading a checkpoint to capture a few movies at that point in training.

Below are some videos of this example mid-way through training and at convergence.

![midway](./docs/midway.gif)
![column](./docs/column.gif)
