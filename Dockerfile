# docker run -it --rm --gpus all --network host growth python run_configurations/minecraft/run.py
FROM rayproject/ray-ml:2.1.0-gpu

WORKDIR /ray

RUN git clone https://github.com/real-itu/Evocraft-py
RUN pip install grpcio

COPY --chown=ray:users . conditional-growth
RUN sudo chown ray conditional-growth
WORKDIR conditional-growth
RUN pip install -e .

