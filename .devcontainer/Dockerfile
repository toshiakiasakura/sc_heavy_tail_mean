FROM quay.io/jupyter/datascience-notebook:x86_64-julia-1.11.1
# julia-1.7.3 is required for Turing.jl

WORKDIR /workdir
EXPOSE 8888

RUN JULIA_PROJECT=/workdir
ENV DEVCONTAINER=TRUE

# TODO: Activate checkpoint and install packages.
RUN R -e "install.packages(c('socialmixr', 'jsonlite', 'here', 'checkpoint', 'cmoceean'), repos = 'https://cran.ism.ac.jp')"
RUN R -e "install.packages(c('tableone', 'broom'), repos = 'https://cran.ism.ac.jp')"

# Install Node.js 20 and Claude Code CLI
ARG CLAUDE_CODE_VERSION=latest
USER root
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    npm install -g @anthropic-ai/claude-code@${CLAUDE_CODE_VERSION} && \
    mkdir -p /home/jovyan/.julia /home/jovyan/.claude && \
    chown jovyan:users /home/jovyan/.julia /home/jovyan/.claude

USER jovyan

# Install RTK (Rust Token Killer) for token-efficient Claude Code sessions
RUN curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh && \
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> /home/jovyan/.bashrc