FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG INSTALL_TRELLIS=0

ENV TZ=Etc/UTC
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=/opt/conda/bin:${PATH}
ENV HF_HOME=/workspace/.cache/huggingface

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      wget curl ca-certificates git bzip2 \
      build-essential ffmpeg libgl1 libglib2.0-0 libssl-dev \
      pkg-config python3-dev unzip && \
    rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xj -C /usr/local/bin --strip-components=1 bin/micromamba && \
    micromamba shell init -s bash -p ${MAMBA_ROOT_PREFIX}

SHELL ["bash", "-lc"]

COPY models/synchuman/environment.yml /tmp/environment.yml

RUN micromamba create -y -n synchuman -f /tmp/environment.yml && \
    micromamba clean -ya

ENV CONDA_DEFAULT_ENV=synchuman
ENV PATH=/opt/conda/envs/synchuman/bin:${PATH}

WORKDIR /workspace

# Clone code
RUN git clone --depth 1 https://github.com/IGL-HKUST/SyncHuman.git

# Optional: install TRELLIS (as recommended in SyncHuman docs). Toggle with --build-arg INSTALL_TRELLIS=1.
RUN if [ "${INSTALL_TRELLIS}" = "1" ]; then \
      git clone --depth 1 https://github.com/microsoft/TRELLIS.git /workspace/TRELLIS && \
      micromamba run -n synchuman pip install -e /workspace/TRELLIS; \
    fi

# Install API deps
COPY models/synchuman/api_requirements.txt /workspace/SyncHuman/api_requirements.txt
RUN micromamba run -n synchuman pip install -r /workspace/SyncHuman/api_requirements.txt

# Install SyncHuman as editable to ensure imports work cleanly
RUN micromamba run -n synchuman pip install -e /workspace/SyncHuman

# Prefetch weights into the image
RUN cd /workspace/SyncHuman && micromamba run -n synchuman python download.py

# Helper scripts
COPY models/synchuman/run_pipeline.py /workspace/SyncHuman/run_pipeline.py
COPY models/synchuman/api_server.py /workspace/SyncHuman/api_server.py

WORKDIR /workspace/SyncHuman

ENV ATTN_BACKEND=flash_attn
ENV SPARSE_ATTN_BACKEND=flash_attn

EXPOSE 8000

ENTRYPOINT ["micromamba","run","-n","synchuman","uvicorn","api_server:app","--host","0.0.0.0","--port","8000"]
