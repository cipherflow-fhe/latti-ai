FROM nvidia/cuda:12.6.3-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies + deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    build-essential gcc-12 g++-12 git wget ca-certificates \
    python3.12 python3.12-venv python3.12-dev \
    libhdf5-dev libhdf5-cpp-103 libomp-dev libssl-dev zlib1g-dev libgmp-dev libntl-dev \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Set python3.12 as default python3 and install pip via get-pip.py
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 100 \
    && wget -q https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py \
    && rm get-pip.py

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# CMake 3.28 (apt version too old, HEonGPU requires >= 3.26)
RUN wget -q https://ghfast.top/https://github.com/Kitware/CMake/releases/download/v3.28.3/cmake-3.28.3-linux-x86_64.tar.gz \
    && tar -C /usr/local --strip-components=1 -xzf cmake-3.28.3-linux-x86_64.tar.gz \
    && rm cmake-3.28.3-linux-x86_64.tar.gz

# Go 1.24.0
RUN wget -q https://golang.google.cn/dl/go1.24.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz \
    && rm go1.24.0.linux-amd64.tar.gz
ENV PATH="/usr/local/go/bin:${PATH}"

# Python dependencies (full torch, using China mirror)
COPY training/requirements.txt /tmp/requirements.txt
RUN python3 -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt
