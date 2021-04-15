FROM nvcr.io/nvidia/cuda:10.2-devel-ubuntu18.04

WORKDIR "/root"

RUN apt-get update
RUN apt-get remove -y --purge cmake
RUN apt-get install -y git libboost-all-dev wget screen vim

# Install Miniconda
ENV PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda install python=3.7

Build voxcraft-sim
RUN conda install -c anaconda cmake==3.12.0
RUN git clone https://github.com/voxcraft/voxcraft-sim.git \
    && cd voxcraft-sim && mkdir build && cd build && cmake .. && make -j 10

# Add project requirements
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 rsync libgl1-mesa-dev xvfb
RUN pip install ray==1.1.0 torch==1.7.1 gym==0.17.3 pandas==1.1.5
RUN pip install pyvista==0.27.4 dm-tree==0.1.5 lxml==4.6.2 pytest==6.2.1 matplotlib==3.3.3
# It's necessary to install rllib after the previous line.
RUN pip install ray[rllib]
RUN pip install vtk==8.1.2

COPY . conditional-growth
WORKDIR conditional-growth
RUN pip install -e .
