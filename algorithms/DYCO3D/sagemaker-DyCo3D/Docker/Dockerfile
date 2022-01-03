#FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
#FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

# Install system dependencies
#ENV DEBIAN_FRONTEND=noninteractive 
RUN apt-get update 
RUN  apt-get install -y wget build-essential git
RUN ln -s /usr/lib/x86_64-linux-gnu/libmpfr.so.6 /usr/lib/x86_64-linux-gnu/libmpfr.so.4

# Install python dependencies
ENV PYTHON_VERSION=3.7
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.12-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
	conda install -y -c bioconda google-sparsehash && \
	conda install -y libboost && \ 
	conda install -y -c daleydeng gcc-5
    #conda install -y scipy pillow gunicorn==19.9.0 && \
    #conda install -y gevent flask matplotlib && \
    #conda install -y -c conda-forge opencv

RUN conda install -y scipy

ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

RUN pip install torch==1.1 cmake>=3.13.2 plyfile tensorboardX pyyaml

RUN apt-get install -y libboost-dev

COPY lib /opt/program/lib

WORKDIR /opt/program/lib/spconv
RUN rm -rf build
RUN python setup.py bdist_wheel
WORKDIR /opt/program/lib/spconv/dist
RUN pip install *.whl

#RUN conda install -y -c bioconda google-sparsehash 
RUN apt-get install -y libsparsehash-dev
WORKDIR  /opt/program/lib/pointgroup_ops
RUN rm -rf build
RUN python setup.py develop

#RUN chmod +x /opt/program/train
#RUN chmod +x /opt/program/serve

COPY sagemaker-DyCo3D/scripts /opt/program
COPY DyCo3D/ /opt/program

WORKDIR /opt/program



