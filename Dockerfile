# Note this Dockerfile is designed for testing rather than end-user use.
# That being said it could be useful as a basis for the latter if required.

# See https://github.com/NVIDIA/nvidia-docker for instructions for ensuring GPU compatibility.
FROM nvidia/cuda:10.0-base-ubuntu18.04

RUN apt update

# Install things we want for cli/installation
RUN apt-get install -y apt-utils git wget

# Install miniconda
WORKDIR /root/install
RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /root/miniconda.sh
RUN bash /root/miniconda.sh -b -p /root/miniconda

WORKDIR /root/pykilosort

# Copy these into build context for install.
# Don't add more files yet to preserve the cached layer.
COPY pyks2.yml ./
COPY test_requirements.txt ./

# Be great to do better at caching this.
RUN eval "$(/root/miniconda/bin/conda shell.bash hook)" && conda init && \
    conda env create -f pyks2.yml && \
    conda activate pyks2 && \
    pip install -r test_requirements.txt

COPY . ./