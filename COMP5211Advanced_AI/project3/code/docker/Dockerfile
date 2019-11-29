ARG IMAGE_NAME
FROM nvidia/cuda:8.0-cudnn5-runtime-ubuntu16.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list

ENV CUDNN_VERSION 7.2.1.38
LABEL com.nvidia.cudnn.version="${CUDNN_VERSION}"

RUN DEBIAN_FRONTEND=noninteractive apt-get update && apt-get install -y --no-install-recommends \
            libcudnn7=$CUDNN_VERSION-1+cuda8.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*


# install basics
RUN apt-get update -y \
 && apt-get install -y apt-utils git curl ca-certificates bzip2 cmake tree htop bmon iotop g++

# Install Miniconda
RUN curl -so /miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x /miniconda.sh \
 && /miniconda.sh -b -p /miniconda \
 && rm /miniconda.sh

ENV PATH=/miniconda/bin:$PATH

# Create a Python 3.7 environment
RUN /miniconda/bin/conda install -y conda-build \
 && /miniconda/bin/conda create -y --name py37 python=3.7 \
 && /miniconda/bin/conda clean -ya

ENV CONDA_DEFAULT_ENV=py37
ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
ENV PATH=$CONDA_PREFIX/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

RUN conda install -y ipython
RUN pip install ninja yacs cython matplotlib


#install cv2
RUN apt-get install python-opencv -y
RUN pip install opencv-python

#install scipy
RUN pip install scipy

#install vim
RUN apt-get install vim -y

#install tqdm
RUN pip install tqdm

#install requests, bs4
RUN pip install requests
RUN pip install bs4
RUN pip install imageio
RUN pip install hickle

ENV DEBIAN_FRONTEND noninteractive

#install keras
RUN pip install tensorflow-gpu==1.14

#install Bert
RUN pip install bert-serving-server  # server
RUN pip install bert-serving-client  # client, independent of `bert-serving-server`

