# ------------------------------------------------------------------
# Base image (official Debian)
# ------------------------------------------------------------------
FROM debian:12-slim

ENV DEBIAN_FRONTEND=noninteractive
ENV MAMBA_ROOT_PREFIX=/opt/conda
ENV PATH=$MAMBA_ROOT_PREFIX/bin:$PATH

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
# Create non-root user
# ------------------------------------------------------------------
RUN useradd -m -u 1000 -s /bin/bash coder
USER coder
WORKDIR /home/coder/app

# ------------------------------------------------------------------
# Copy environment and install dependencies
# ------------------------------------------------------------------
COPY --chown=coder:coder environment.yml .
COPY --chown=coder:coder . .

# Create environment
RUN mamba env create -f environment.yml && \
    mamba clean -a -y

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