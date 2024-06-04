FROM debian:bookworm

ENV INST 'env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends'
ENV PIPINST 'python3 -m pip install --no-cache-dir --upgrade'

RUN apt-get update && $INST \
    autoconf \
    automake \
    build-essential \
    ccache \
    clang \
    clang-format \
    clang-tidy \
    cmake \
    coreutils \
    curl \
    device-tree-compiler \
    dfu-util \
    expect \
    fd-find \
    ffmpeg \
    file \
    fonts-lato \
    g++ \
    gcc \
    gcc-multilib \
    git \
    git-lfs \
    g++-multilib \
    gperf \
    gtk-sharp2 \
    jq \
    libeigen3-dev \
    libevent-dev \
    libftdi1-dev \
    libglfw3 \
    libglfw3-dev \
    libglib2.0-0 \
    libgomp1 \
    libgtk2.0-dev \
    libjson-c-dev \
    libopenblas-dev \
    libprotoc-dev \
    libpython3-dev \
    libtinfo5 \
    libtinfo-dev \
    libtool \
    libusb-1.0.0-dev \
    llvm \
    llvm-dev \
    make \
    mono-complete \
    mono-devel \
    ninja-build \
    openocd \
    picocom \
    policykit-1 \
    procps \
    protobuf-compiler \
    python3 \
    python3-dev \
    python3-opencv \
    python3-pip \
    python3-venv \
    software-properties-common \
    sudo \
    tar \
    tmux \
    uml-utilities \
    unzip \
    verilator \
    vim \
    wget \
    xxd \
    xz-utils \
    zip \
    && apt-get autoremove && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN rm /usr/lib/python3.11/EXTERNALLY-MANAGED
RUN $PIPINST pip setuptools PyYAML cmake flit_core

RUN $PIPINST \
    amaranth-yosys \
    cffi \
    cloudpickle \
    flake8 \
    numpy \
    opencv-python \
    pre-commit \
    psutil \
    pydocstyle \
    pyelftools \
    pyserial \
    pytest \
    tornado \
    tqdm \
    west \
    ruff

COPY requirements.txt /tmp/requirements.txt

RUN $PIPINST -r /tmp/requirements.txt

RUN cd / \
    && wget https://static.dev.sifive.com/dev-tools/freedom-tools/v2020.08/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tar.gz \
    && tar xvzf /riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tar.gz \
    && rm /riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14.tar.gz

ENV PATH="${PATH}:/riscv64-unknown-elf-gcc-10.1.0-2020.08.2-x86_64-linux-ubuntu14/bin"
