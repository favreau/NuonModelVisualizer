
# The Blue Brain BioExplorer is a tool for scientists to extract and analyse
# scientific data from visualization
#
# Copyright 2020-2023 Blue BrainProject / EPFL
# 
# This program is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program.  If not, see <https://www.gnu.org/licenses/>.

FROM python:3.10-slim as builder
LABEL maintainer="cyrille.favreau@epfl.ch"

ARG DIST_PATH=/app

RUN apt-get update && \
    apt-get install -y --no-install-recommends wget git g++ gcc binutils libxpm-dev libxft-dev libxext-dev python3 libssl-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# --------------------------------------------------------------------------------
# Install ROOT
# --------------------------------------------------------------------------------
ARG ROOT_VERSION=6.30.04
ARG ROOT_DIR=root_v${ROOT_VERSION}.Linux-ubuntu22.04-x86_64-gcc11.4
ARG ROOT_PATH=/app/$ROOT_DIR

RUN mkdir -p ${ROOT_PATH} \
   && wget --no-verbose https://root.cern/download/${ROOT_DIR}.tar.gz \
   && tar zxvf ${ROOT_DIR}.tar.gz -C ${ROOT_PATH} --strip-components=1 \
   && rm ${ROOT_DIR}.tar.gz

# Add ROOT bin to the PATH
ENV PATH $PATH:${ROOT_PATH}/bin
ENV PYTHONPATH $PYTHONPATH:${ROOT_PATH}/lib

WORKDIR /app
ADD . /app/NuonModelVisualizer

RUN cd /app/NuonModelVisualizer && \
    python3 -m venv /app/NuonModelVisualizer/env \
    && . /app/NuonModelVisualizer/env/bin/activate \
    && python3 -m pip install --upgrade pip --use-deprecated=legacy-resolver \
    && python3 -m pip install wheel \
    && python3 -m pip install -e .

# --------------------------------------------------------------------------------
# Install BioExplorer
# https://github.com/favreau/BioExplorer
# --------------------------------------------------------------------------------
ARG BIOEXPLORER_SRC=/app/bioexplorer

RUN mkdir -p ${BIOEXPLORER_SRC} \
   && git clone https://github.com/favreau/BioExplorer.git ${BIOEXPLORER_SRC} \
   && cd ${BIOEXPLORER_SRC}/bioexplorer/pythonsdk \
   && . /app/NuonModelVisualizer/env/bin/activate \
   && pip install . --use-deprecated=legacy-resolver \
   && cd /app \
   && rm -rf ${BIOEXPLORER_SRC}

# --------------------------------------------------------------------------------
# Install Rockets (from favreau for Python 3.10)
# https://github.com/favreau/Rockets
# --------------------------------------------------------------------------------
ARG ROCKETS_SRC=/app/rockets

RUN mkdir -p ${ROCKETS_SRC} \
   && git clone https://github.com/favreau/Rockets.git ${ROCKETS_SRC} \
   && cd ${ROCKETS_SRC}/python \
   && . /app/NuonModelVisualizer/env/bin/activate \
   && pip install . --use-deprecated=legacy-resolver \
   && pip install numpy==1.22.4 \
   && cd /app \
   && rm -rf ${ROCKETS_SRC}

RUN rm -rf /tmp/*

ENV PATH /app/NuonModelVisualizer:$PATH

RUN chmod +x /app/NuonModelVisualizer/nuon_model_visualizer_python_sdk

# Expose a port from the container
# For more ports, use the `--expose` flag when running the container,
# see https://docs.docker.com/engine/reference/run/#expose-incoming-ports for docs.
EXPOSE 8888

ENTRYPOINT ["nuon_model_visualizer_python_sdk"]
