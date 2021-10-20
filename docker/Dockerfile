FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# nethack dependencies
RUN apt-get update
RUN apt-get install -y build-essential autoconf libtool pkg-config python3-dev python3-pip python3-numpy git flex bison libbz2-dev
RUN git clone https://github.com/Kitware/CMake/; cd CMake

# add code
WORKDIR /opt/silg
RUN mkdir -p /opt/silg

# install cmake
RUN mkdir -p ext
RUN apt-get install -y libssl-dev
RUN git clone https://github.com/Kitware/CMake ext/cmake
RUN cd ext/cmake && ./bootstrap
RUN cd ext/cmake && make
RUN cd ext/cmake && make install

# alfworld dependencies
RUN apt-get install -y curl libgl1 wget

# install environments
ADD install_envs.sh /opt/silg/
RUN bash install_envs.sh

# add SILG package
ADD core core
ADD model model
ADD silg silg
ADD *.py *.txt *.md .

# install SILG package
RUN pip install -r requirements.txt
RUN pip install -e .
