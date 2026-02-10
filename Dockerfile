# ============================================================================
# Encrypted Inference Development Docker Image (CPU)
# ============================================================================
# Development environment for privacy-preserving AI inference.
#
# Build:
#   docker build -t latti-ai:dev .
#
# Run:
#   docker run -it latti-ai:dev
# ============================================================================

FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies and add deadsnakes PPA for Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    build-essential \
    gcc-12 \
    g++-12 \
    cmake \
    git \
    wget \
    ca-certificates \
    libhdf5-dev \
    libhdf5-cpp-103 \
    libomp-dev \
    libssl-dev \
    zlib1g-dev \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Set GCC 12 as default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-12 100

# Install Go 1.24.0 (required for Lattigo cryptography library)
RUN wget -q https://go.dev/dl/go1.24.0.linux-amd64.tar.gz \
    && tar -C /usr/local -xzf go1.24.0.linux-amd64.tar.gz \
    && rm go1.24.0.linux-amd64.tar.gz

ENV PATH="/usr/local/go/bin:${PATH}"
ENV GOPATH="/root/go"

# Copy source code
WORKDIR /workspace
COPY . /workspace

# Initialize git submodules
RUN git submodule update --init --recursive

# Create Python virtual environment and install dependencies
RUN python3.12 -m venv /workspace/venv \
    && /workspace/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /workspace/venv/bin/pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && grep -v '^torch' /workspace/training/requirements.txt \
       | /workspace/venv/bin/pip install --no-cache-dir -r /dev/stdin

# Create build directory
RUN mkdir -p /workspace/build

# Set environment variables
ENV PATH="/workspace/venv/bin:${PATH}"
ENV VIRTUAL_ENV="/workspace/venv"

WORKDIR /workspace

# Default command
CMD ["/bin/bash"]
