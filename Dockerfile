# ------------------------------------------------------
# Base image: CUDA 12.4.1, Ubuntu 22.04
# ------------------------------------------------------
ARG CUDA_IMAGE="12.4.1-devel-ubuntu22.04"
FROM nvidia/cuda:${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

# ------------------------------------------------------
# System deps: Python, build tools, ODBC, curl, gnupg, Node
# ------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        wget \
        git \
        build-essential \
        curl \
        gnupg \
        unixodbc \
        unixodbc-dev && \
    rm -rf /var/lib/apt/lists/*

# MSSQL ODBC driver
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - && \
    curl https://packages.microsoft.com/config/ubuntu/22.04/prod.list \
        > /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update && ACCEPT_EULA=Y apt-get install -y --no-install-recommends \
        msodbcsql17 && \
    rm -rf /var/lib/apt/lists/*

# Node.js for NeurowebUI
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------
# Common env
# ------------------------------------------------------
ENV HOST=0.0.0.0
ENV CUDA_DOCKER_ARCH=all
ENV GGML_CUDA=1
ENV IS_DOCKER=1
ENV DEFAULT_LLM_MODEL=unsloth

WORKDIR /app

# ------------------------------------------------------
# install deps
COPY requirements.txt ./
RUN python3 -m pip install -r requirements.txt

# install torch and other llm deps
RUN python3 -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126
RUN python3 -m pip install transformers>="4.56,<5.0" sentence-transformers>="5.1,<6.0" bitsandbytes>="0.48,<0.49" 
RUN python3 -m pip install unsloth=="2025.8.10" accelerate>="1.10.1,<2.0"

# ------------------------------------------------------
# 2) Now copy the rest of your project
#    This won't trigger re-install of deps unless you touch install.py.
# ------------------------------------------------------
COPY . .

# ------------------------------------------------------
# 3) Run the CLI from source (no need for pip install -e .)
# ------------------------------------------------------
# Importing neurosurfer from /app works because it's on PYTHONPATH by default
CMD ["python3", "-m", "neurosurfer.cli.main", "serve"]
