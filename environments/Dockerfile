FROM debian:bullseye

ENV INST 'env DEBIAN_FRONTEND=noninteractive apt-get install -y'
ENV PIPINST 'python3 -m pip install --no-cache-dir --upgrade'

RUN apt-get update && INST \
    build-essential \
    curl \
    fd-find \
    g++ \
    gcc \
    git \
    libglfw3 \
    libglfw3-dev \
    libgomp1 \
    libopenblas-dev \
    libprotoc-dev \
    libtinfo-dev \
    llvm \
    llvm-dev \
    protobuf-compiler \
    python3 \
    python3-dev \
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
    cloudpickle \
    cmake \
    flake8 \
    ipython \
    matplotlib \
    netron \
    numpy \
    onnx \
    opencv-python \
    psutil \
    ptipython \
    pytest \
    scikit-learn \
    tensorflow \
    tensorflow-addons \
    tensorflow-model-optimization \
    tflite \
    tornado \
    tqdm \
    xgboost

RUN $PIPINST git+https://github.com/onnx/onnx-tensorflow.git@v1.9.0
RUN $PIPINST git+https://github.com/onnx/tensorflow-onnx.git@v1.9.2

RUN git clone --recursive https://github.com/apache/tvm.git -b v0.9.dev0 /tvm && \
    cd /tvm && mkdir build && cd build && \
    mv /tvm-config.cmake /tvm/build/config.cmake && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j`nproc` && \
    make install && \
    ldconfig && \
    $PIPINST -e /tvm/python && \
    cd / && \
    rm -rf /tvm/build

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
