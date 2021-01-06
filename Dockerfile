FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR "/root"

RUN apt-get update
RUN apt-get remove -y --purge cmake
RUN apt-get install -y git libboost-all-dev wget screen 

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 

# Build voxcraft-sim
RUN conda install -c anaconda cmake
RUN git clone https://github.com/voxcraft/voxcraft-sim.git \
    && cd voxcraft-sim && mkdir build && cd build && cmake .. && make -j 10

# Add project requirements
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 rsync
RUN pip install ray torch gym dm-tree pandas lxml
# It's necessary to install rllib outside of the previous line.
RUN pip install ray[rllib]

COPY . conditional-growth
WORKDIR conditional-growth
RUN pip install -e .
