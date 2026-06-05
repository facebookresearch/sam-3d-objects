# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG USERNAME=developer
ARG USER_UID=1000
ARG USER_GID=1000
ARG EXTRA_APT_PACKAGES=""
ARG TORCH_CUDA_ARCH_LIST="86"

# Base OS setup
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    ffmpeg \
    git \
    git-lfs \
    libffi-dev \
    libglib2.0-0 \
    libgl1 \
    libjpeg-dev \
    libomp-dev \
    libopenexr-dev \
    libpng-dev \
    libsm6 \
    libssl-dev \
    libtiff-dev \
    libx11-6 \
    libxext6 \
    libxrender1 \
    locales \
    ninja-build \
    pciutils \
    pkg-config \
    python3 \
    python3-dev \
    python3-venv \
    sudo \
    unzip \
    vim \
    wget \
    && rm -rf /var/lib/apt/lists/*

RUN if [ -n "${EXTRA_APT_PACKAGES}" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends ${EXTRA_APT_PACKAGES} && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# Configure UTF-8 locale
RUN locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# CUDA 12.9 toolchain paths are required during dependency builds
ENV CUDA_HOME=/usr/local/cuda-12.9 \
    PATH=${CUDA_HOME}/bin:${PATH} \
    LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

ENV TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}

# Install uv (https://github.com/astral-sh/uv) system-wide
RUN curl -LsSf https://astral.sh/uv/install.sh | sh \
    && install -Dm755 /root/.local/bin/uv /usr/local/bin/uv

# Create a non-root user that matches the host UID/GID for seamless volume access
RUN set -eux; \
    if ! getent group "${USER_GID}" >/dev/null; then \
    groupadd --gid "${USER_GID}" "${USERNAME}"; \
    fi; \
    EXISTING_USER="$(getent passwd "${USER_UID}" | cut -d: -f1 || true)"; \
    if [ -n "${EXISTING_USER}" ] && [ "${EXISTING_USER}" != "${USERNAME}" ]; then \
    usermod -l "${USERNAME}" "${EXISTING_USER}"; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
    usermod -u "${USER_UID}" -g "${USER_GID}" "${USERNAME}"; \
    else \
    useradd --uid "${USER_UID}" --gid "${USER_GID}" -m "${USERNAME}"; \
    fi; \
    usermod -d /home/${USERNAME} -m "${USERNAME}"; \
    usermod -aG sudo "${USERNAME}"; \
    echo "%${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME}; \
    chmod 440 /etc/sudoers.d/${USERNAME}; \
    mkdir -p /workspace /opt/sam3d /opt/uv-python; \
    chown -R ${USERNAME}:${USER_GID} /workspace /opt/sam3d /opt/uv-python

USER ${USERNAME}
WORKDIR /opt/sam3d

ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python

# Copy the minimum project files required to materialize the environment
COPY --chown=${USERNAME}:${USER_GID} pyproject.toml uv.lock README.md ./ 
COPY --chown=${USERNAME}:${USER_GID} patching ./patching
COPY --chown=${USERNAME}:${USER_GID} sam3d_objects ./sam3d_objects

# Provision the Python 3.11 environment with uv and patch Hydra
RUN uv python install 3.11 \
    && uv sync --frozen --python 3.11 --no-editable \
    && .venv/bin/python patching/hydra

ENV VIRTUAL_ENV=/opt/sam3d/.venv
ENV PATH=${VIRTUAL_ENV}/bin:${PATH}

WORKDIR /workspace

# Default shell
CMD ["bash"]
