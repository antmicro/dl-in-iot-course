FROM debian:bookworm

ENV INST 'env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends'
ENV PIPINST 'python3 -m pip install --no-cache-dir --upgrade'

RUN apt-get update && $INST \
    build-essential \
    cmake \
    curl \
    fd-find \
    ffmpeg \
    fonts-lato \
    g++ \
    gcc \
    git \
    git-lfs \
    libeigen3-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libgomp1 \
    libopenblas-dev \
    libprotoc-dev \
    libpython3-dev \
    libtinfo5 \
    libtinfo-dev \
    llvm \
    llvm-dev \
    mono-complete \
    procps \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-opencv \
    python3-pip \
    python3-venv \
    wget \
    zip \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED
RUN $PIPINST pip setuptools PyYAML cmake flit_core

RUN $PIPINST \
    flake8 \
    pre-commit \
    pydocstyle \
    ruff

COPY requirements.txt /tmp/requirements.txt

RUN $PIPINST -r /tmp/requirements.txt
