#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Build based on the official PyTorch docker image available on docker hub:
# https://hub.docker.com/r/pytorch/pytorch/tags
# Select an image from the website which most closely matches the
# versions used to build your model.
# Here we're using PyTorch 1.7.0 with CUDA 11.0
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
#FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

# Add metadata
LABEL maintainer="siharris@au1.ibm.com"
LABEL version="0.1"
LABEL description="Docker image for ObjectNet AI Challenge."

ARG MODEL_CHECKPOINT
ARG MODEL_CLASS_NAME
ENV MODEL_CLASS_NAME ${MODEL_CLASS_NAME}
ENV MODEL_PATH "/workspace/model/"${MODEL_CHECKPOINT}

RUN echo "Using pre-built model: $MODEL_PATH"

# Set working directory# Add metadata
WORKDIR /workspace

# Copy (recursively) all files from the current directory into the
# image at /workspace
COPY . /workspace

# Install python dependencies defined in requirements.txt
RUN pip install -r requirements.txt

# Define the command to execute when the container is run
ENTRYPOINT python objectnet_eval.py /input /output/predictions.csv $MODEL_CLASS_NAME $MODEL_PATH 
