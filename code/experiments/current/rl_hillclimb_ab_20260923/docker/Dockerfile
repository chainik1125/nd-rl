FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends tmux git curl ca-certificates procps less jq && rm -rf /var/lib/apt/lists/*
# Claude Code CLI as a system-wide binary (the agent's home is a bind mount at run time)
RUN curl -fsSL https://claude.ai/install.sh | bash && cp -L /root/.local/bin/claude /usr/local/bin/claude && chmod 755 /usr/local/bin/claude && /usr/local/bin/claude --version
RUN groupadd -g 10001 agent && useradd -m -u 10001 -g 10001 -s /bin/bash agent
USER agent
WORKDIR /work
