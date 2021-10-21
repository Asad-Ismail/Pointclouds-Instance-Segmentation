#FROM nvidia/cuda:11.2.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04
ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update && apt-get install -y --no-install-recommends \
		ca-certificates \
		curl \
		netbase \
		wget \
       build-essential \
        libopencv-dev\
        libopenblas-dev \
        nginx

ENV PYTHON_VERSION=3.7

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=$PYTHON_VERSION 


#COPY ./tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl /opt/
#RUN pip install /opt/tensorflow-2.3.0-cp38-cp38-linux_x86_64.whl 
RUN pip install scipy pillow gunicorn==19.9.0 gevent flask

RUN pip install torch==1.1 cmake>=3.13.2 plyfile tensorboardX pyyaml scipy open3d

#RUN pip install bioconda google-sparsehash 

RUN conda install -y  -c bioconda google-sparsehash 

RUN conda install -y  libboost
RUN conda install -y -c daleydeng gcc-5 # need gcc-5.4 for sparseconv

COPY ./DyCo3D/ /opt/program/DyCo3D/

RUN  apt-get install -y libboost-all-dev
WORKDIR /opt/program/DyCo3D/lib/spconv

RUN rm -rf build
RUN python setup.py bdist_wheel
WORKDIR /opt/program/DyCo3D/lib/spconv/dist
RUN pip install spconv-1.0-cp37-cp37m-linux_x86_64.whl

WORKDIR /opt/program/DyCo3D/lib/pointgroup_ops
RUN python setup.py develop

#ENV PYTHONUNBUFFERED=TRUE
#ENV PYTHONDONTWRITEBYTECODE=TRUE
#ENV PATH="/opt/program:${PATH}"



#RUN chmod +x /opt/program/train
#RUN chmod +x /opt/program/serve

#WORKDIR /opt/program/Dyco3D
WORKDIR /opt/program/DyCo3D/
RUN python train.py




