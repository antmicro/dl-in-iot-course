FROM archlinux:latest

ENV PIPINST 'python -m pip install --no-cache-dir --upgrade'

RUN pacman --noconfirm -Syu
RUN pacman --noconfirm -Sy \
    blas \
    cblas \
    clang \
    cmake \
    curl \
    fd \
    flake8 \
    gcc \
    git \
    gtest \
    ipython \
    llvm \
    make \
    protobuf \
    pybind11 \
    python-cffi \
    python-cloudpickle \
    python-jinja \
    python-matplotlib \
    python-numpy \
    python-opencv \
    python-pip \
    python-psutil \
    python-pytest \
    python-pytorch \
    python-scikit-learn \
    python-setuptools \
    python-tornado \
    python-tqdm \
    tmux \
    vim \
    wget \
    zip

RUN $PIPINST \
    cloudpickle \
    flatbuffers \
    jupyterlab \
    netron \
    nni \
    notebook \
    onnx \
    ptipython \
    tensorflow==2.10 \
    tensorflow-addons==0.18.0 \
    tensorflow-model-optimization==0.7.2 \
    tensorflow-probability==0.18.0 \
    tflite \
    torchvision==0.13.1 \
    "xgboost>=1.1.0,<1.6.0"

RUN $PIPINST git+https://github.com/onnx/onnx-tensorflow.git@v1.10.0
RUN $PIPINST git+https://github.com/onnx/tensorflow-onnx.git@v1.13.0

COPY cpu-config.cmake /tvm-config.cmake
RUN cd / && git clone --recursive https://github.com/apache/tvm.git -b v0.10.0 /tvm && \
    cd /tvm && mkdir build && cd build && \
    mv /tvm-config.cmake /tvm/build/config.cmake && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j`nproc` && \
    make install && \
    ldconfig && \
    CONDA_BUILD=yes $PIPINST -e /tvm/python && \
    cd / && \
    rm -rf /tvm/build

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
