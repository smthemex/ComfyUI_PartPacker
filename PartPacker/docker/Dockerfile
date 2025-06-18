# get the development image from nvidia cuda 12.4 (using devel for full CUDA toolkit)
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

LABEL name="partpacker" maintainer="partpacker"

# create workspace folder and set it as working directory
RUN mkdir -p /workspace
WORKDIR /workspace

# update package lists and install essential packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    wget \
    vim \
    libegl1-mesa-dev \
    libglib2.0-0 \
    unzip \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install additional graphics and rendering dependencies
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    pkg-config \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    libgles2 \
    libglvnd-dev \
    libgl1-mesa-dev \
    libegl1-mesa-dev \
    libgles2-mesa-dev \
    cmake \
    mesa-utils-extra \
    libxrender1 \
    libxi6 \
    libgconf-2-4 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
ENV PYOPENGL_PLATFORM=egl

# Set CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;8.9;9.0"

# install conda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x Miniconda3-latest-Linux-x86_64.sh && \
    ./Miniconda3-latest-Linux-x86_64.sh -b -p /workspace/miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# update PATH environment variable
ENV PATH="/workspace/miniconda3/bin:${PATH}"

# initialize conda
RUN conda init bash

# create and activate conda environment
RUN conda create -n partpacker python=3.10 && echo "source activate partpacker" > ~/.bashrc
ENV PATH="/workspace/miniconda3/envs/partpacker/bin:${PATH}"

# Set conda to always auto-approve
RUN conda config --set always_yes true

# Install essential conda packages
RUN conda install Ninja
RUN conda install cuda -c nvidia/label/cuda-12.4.1 -y

# Update libstdcxx-ng to fix compatibility issues
RUN conda install -c conda-forge libstdcxx-ng -y

# Install PyTorch with CUDA support
RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Clone PartPacker repository
RUN git clone https://github.com/NVlabs/PartPacker.git

# Set working directory to the cloned repository
WORKDIR /workspace/PartPacker

# Clean up requirements.txt to remove invalid pip options
RUN sed -i 's/ --no-build-isolation//g' requirements.txt && \
    sed -i 's/--no-build-isolation//g' requirements.txt

# Install Python dependencies
RUN pip install -r requirements.txt

# Install transformers
RUN pip install transformers

# Modify app.py to enable share=True for Gradio
RUN sed -i 's/block\.launch()/block.launch(share=True)/g' app.py

# Create pretrained models directory and download models
RUN mkdir -p pretrained && \
    cd pretrained && \
    wget https://huggingface.co/nvidia/PartPacker/resolve/main/vae.pt && \
    wget https://huggingface.co/nvidia/PartPacker/resolve/main/flow.pt

# Set global library paths to ensure proper linking at runtime
ENV LD_LIBRARY_PATH="/workspace/miniconda3/envs/partpacker/lib:${LD_LIBRARY_PATH}"

# Activate conda environment by default
RUN echo "conda activate partpacker" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Cleanup
RUN apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    conda clean -a -y

# Expose port for Gradio app
EXPOSE 7860

# Set default command to bash
CMD ["/bin/bash"]