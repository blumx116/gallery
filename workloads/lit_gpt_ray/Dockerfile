ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.02-py3
FROM ${BASE_IMAGE} as base-image

# Ensure apt-get won't prompt for selecting options
ENV DEBIAN_FRONTEND=noninteractive
# libavdevice-dev rerquired for latest torchaudio
RUN apt-get update && \
  apt-get upgrade -y && \
  apt-get install -y \
  libsndfile1 sox \
  libfreetype6 \
  swig \
  ffmpeg \
  tcpdump \ 
  iproute2 \
  libavdevice-dev && \
  rm -rf /var/lib/apt/lists/*

# Prerequisite for removing GCSFuse dependency
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] http://packages.cloud.google.com/apt cloud-sdk main" | \
  tee -a /etc/apt/sources.list.d/google-cloud-sdk.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
  apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - \
  && apt-get update -y && apt-get install google-cloud-cli -y

WORKDIR /workspace/

COPY workloads/lit_gpt_ray/pretrain /workspace/pretrain
COPY workloads/lit_gpt_ray/scripts /workspace/pretrain/scripts
COPY workloads/lit_gpt_ray/utilities /workspace/pretrain/utilities
COPY gallery /workspace/pretrain/gallery
COPY gallery/upload_csv.py /workspace/pretrain

RUN pip install --upgrade pip

# Install Lit-GPT and dependencies
# RUN git clone https://github.com/Lightning-AI/lit-gpt.git
# WORKDIR "/workspace/lit-gpt"
# RUN git checkout 44c3c58b759fa0903ab31ed8863a66c157d5ccd9
RUN pip install -r https://raw.githubusercontent.com/Lightning-AI/litgpt/44c3c58b759fa0903ab31ed8863a66c157d5ccd9/requirements-all.txt

RUN pip install git+https://github.com/Lightning-AI/lit-gpt.git@44c3c58b759fa0903ab31ed8863a66c157d5ccd9

# Install Ray
RUN pip install "ray[default]==2.10.0"
RUN pip install "ray[serve]==2.10.0"

RUN pip install  torch==2.2.2 tokenizers sentencepiece ujson \
  torchvision nvidia-dlprof-pytorch-nvtx nvidia-pyindex nvidia-dlprof \
  tokenizers sentencepiece ujson nvidia-nccl-cu12 jsonargparse csvkit \
  pandas google-cloud-bigquery pyarrow>=3.0.0 db-dtypes streamlit


# RUN git clone  -b tejasn/litgpt https://github.com/blumx116/gallery.git@
# WORKDIR /workspace/gallery
# RUN pip install -r requirements.in
# RUN pip install git+https://github.com/blumx116/gallery.git@tejasn/litgpt


# Check install
RUN python -c "from lit_gpt.model import GPT, Block, Config" && \
  python -c "import lightning as L" && \
  python -c "from lightning.fabric.strategies import FSDPStrategy" && \
  python -c "import ray; ray.init()" 

