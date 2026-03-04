# ------------------------------------------------------------------
# Base image (official Debian)
# ------------------------------------------------------------------
FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

# Prevent VSCodium WSL prompt
ENV DONT_PROMPT_WSL_INSTALL=1

# ------------------------------------------------------------------
# System dependencies
# ------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    git \
    bzip2 \
    build-essential \
    wget \
    tini \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------
# Install Miniforge (includes mamba, conda-forge only)
# ------------------------------------------------------------------
RUN wget -qO /tmp/miniforge.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh && \
    bash /tmp/miniforge.sh -b -p $MAMBA_ROOT_PREFIX && \
    rm /tmp/miniforge.sh

# Ensure mamba is available
RUN mamba --version

# ------------------------------------------------------------------
# Install VS Codium (official repository)
# ------------------------------------------------------------------
RUN curl -fsSL https://gitlab.com/paulcarroty/vscodium-deb-rpm-repo/raw/master/pub.gpg \
    | gpg --dearmor -o /usr/share/keyrings/vscodium.gpg && \
    echo "deb [ signed-by=/usr/share/keyrings/vscodium.gpg ] \
    https://download.vscodium.com/debs vscodium main" \
    > /etc/apt/sources.list.d/vscodium.list && \
    apt-get update && \
    apt-get install -y codium && \
    rm -rf /var/lib/apt/lists/*

# Make sure `code` exists in PATH (required by your CMD)
RUN ln -s /usr/bin/codium /usr/bin/code

# ------------------------------------------------------------------
# Install dependencies
# ------------------------------------------------------------------

# Create non-root user
RUN useradd -m -u 1000 -s /bin/bash coder

WORKDIR /home/coder
RUN git clone -b new_renku --single-branch https://github.com/lpego/mzb-suite.git
WORKDIR /home/coder/mzb-suite

# Create environment as root
RUN mamba env create -y -f environment.yml && \
    mamba clean -a -y

# Make conda readable/executable for non-root user
RUN chown -R coder:coder /home/coder

# Switch to non-root user
USER coder

# Activate environment automatically
ENV CONDA_DEFAULT_ENV=mzbsuite
ENV PATH=$MAMBA_ROOT_PREFIX/envs/mzbsuite/bin:$PATH

# ------------------------------------------------------------------
# Expose required port
# ------------------------------------------------------------------
EXPOSE 8888

# ------------------------------------------------------------------
# Required ENTRYPOINT and CMD
# ------------------------------------------------------------------
ENTRYPOINT ["sh", "-c"]

CMD ["code serve-web --server-base-path $RENKU_BASE_URL_PATH/ --without-connection-token --host 0.0.0.0 --port 8888"]