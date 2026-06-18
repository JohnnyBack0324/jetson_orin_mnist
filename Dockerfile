FROM ubuntu:24.04

# ----- 기본 설정 -----
ARG DEBIAN_FRONTEND=noninteractive
ARG PROJECT_NAME
ENV TZ=Asia/Seoul
ENV SHELL=/bin/bash

# ----- APT 미러 교체 및 필수 도구 설치 -----
# Ubuntu 24.04(Noble)는 DEB822 형식(/etc/apt/sources.list.d/ubuntu.sources)을 사용
RUN sed -i 's|http://archive.ubuntu.com|http://mirror.navercorp.com|g' /etc/apt/sources.list.d/ubuntu.sources && \
    sed -i 's|http://security.ubuntu.com|http://mirror.navercorp.com|g' /etc/apt/sources.list.d/ubuntu.sources && \
    apt update && apt upgrade -y && \
    apt install -y --no-install-recommends \
      build-essential wget curl git sudo locales tzdata vim nano less zip unzip tar gzip xz-utils net-tools lsof \
      python3 python3-pip ca-certificates gnupg && \
    apt clean && rm -rf /var/lib/apt/lists/*

# ----- Node.js & npm 설치 (nodejs.org 공식 바이너리 — arm64/x86_64 완벽 호환) -----
# v22.22.2(2026-04-10)는 arm64 tarball에 promise-retry 누락 버그 존재 → 안정 버전 고정
# NODE_LTS: Node.js 22 LTS 검증 버전 (업그레이드 시 이 값만 수정)
ARG NODE_LTS=22.14.0
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then NODE_ARCH="x64"; \
    elif [ "$ARCH" = "aarch64" ]; then NODE_ARCH="arm64"; \
    else echo "Unsupported architecture: $ARCH" && exit 1; fi && \
    curl -fsSL "https://nodejs.org/dist/v${NODE_LTS}/node-v${NODE_LTS}-linux-${NODE_ARCH}.tar.xz" | \
        tar -xJ -C /usr/local --strip-components=1

# ----- 사용자 defuser 생성 및 sudo 권한 부여 -----
RUN useradd -m -s /bin/bash defuser && \
    echo "defuser:defuser" | chpasswd && usermod -aG sudo defuser

# ----- Miniconda 설치 (CPU 아키텍처 자동 감지) -----
RUN ARCH=$(uname -m) && \
    if [ "$ARCH" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$ARCH" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
        echo "Unsupported architecture: $ARCH" && exit 1; \
    fi && \
    wget -q $MINICONDA_URL -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && rm /tmp/miniconda.sh && \
    ln -s /opt/conda/bin/conda /usr/local/bin/conda && \
    echo 'export PATH="/opt/conda/bin:$PATH"' >> /home/defuser/.bashrc && \
    /opt/conda/bin/conda clean -afy

# ----- Poetry 설치 (시스템 전역 경로에 설치) -----
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python3 - && \
    ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry

# ----- defuser 전환 -----
USER defuser
WORKDIR /home/defuser/${PROJECT_NAME}

# ----- Node, Python, Poetry 버전 점검 -----
RUN node -v && npm -v && python3 --version

# ----- 기본 CMD -----
CMD ["/bin/bash"]
