FROM ubuntu:latest

# Install Python and the other dependencies.
RUN \
  apt-get update \
  && apt-get install -y python-pip python3-pip python3-dev git nano vim gdb \
  && apt-get install -y build-essential clang g++ gcc \
  && apt-get install -y libx11-dev swig python-qt4 qt4-dev-tools \
  && apt-get install -y ccache cmake zlib1g-dev libpng-dev libfreetype6-dev libcairo2-dev \
  && apt-get install -y libglu1-mesa-dev freeglut3-dev mesa-common-dev \
  && pip3 install tox black numpy Cython

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV HOME=/data

# Define working directory.
WORKDIR /data

COPY PyAMI PyAMI
RUN pip3 install /data/PyAMI 

COPY requirements.txt .
RUN pip3 install -r requirements.txt

WORKDIR /data/PyBERT

# Define default command.
CMD ["bash"]
