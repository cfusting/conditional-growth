FROM rayproject/ray-ml:1.3.0-gpu

RUN sudo apt-get update
RUN sudo apt-get remove -y --purge cmake
RUN sudo apt-get install -y git libboost-all-dev wget screen vim
RUN sudo apt-get install -y ffmpeg libsm6 libxext6 rsync libgl1-mesa-dev xvfb
RUN pip install pyvista==0.27.4 dm-tree==0.1.5 lxml==4.6.2 pytest==6.2.1 matplotlib==3.3.3 vtk==8.1.2

# pip version has recording bug.
RUN pip uninstall -y gym && git clone https://github.com/openai/gym.git && cd gym && pip install -e . && cd ..

COPY --chown=ray:users . conditional-growth
RUN sudo chown ray conditional-growth
WORKDIR conditional-growth
RUN pip install -e .

