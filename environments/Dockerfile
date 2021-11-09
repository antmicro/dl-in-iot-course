FROM debian:bullseye

ENV INST 'env DEBIAN_FRONTEND=noninteractive apt-get install -y'
ENV PIPINST 'python3 -m pip install --no-cache-dir --upgrade'

RUN apt-get update && INST \
    curl \
    fd-find \
    git \
    python3 \
    python3-pip \
    tmux \
    vim \
    wget \
    zip \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN $PIPINST pip setuptools

RUN $PIPINST \
    Jinja2 \
    cffi \
    flake8 \
    ipython \
    matplotlib \
    netron \
    numpy \
    onnx \
    opencv-python \
    ptipython \
    pytest \
    scikit-learn \
    tensorflow \
    tensorflow-addons \
    tensorflow-model-optimization \
    tqdm

RUN $PIPINST git+https://github.com/onnx/onnx-tensorflow.git@v1.9.0
RUN $PIPINST git+https://github.com/onnx/tensorflow-onnx.git@v1.9.2
