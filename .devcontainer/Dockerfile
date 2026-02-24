FROM quay.io/jupyter/datascience-notebook:x86_64-julia-1.11.1
# julia-1.7.3 is required for Turing.jl

WORKDIR /workdir
EXPOSE 8888

RUN JULIA_PROJECT=/workdir
ENV DEVCONTAINER=TRUE

# TODO: Activate checkpoint and install packages.
RUN R -e "install.packages(c('socialmixr', 'jsonlite', 'here', 'checkpoint', 'cmoceean'), repos = 'https://cran.ism.ac.jp')"
RUN R -e "install.packages(c('tableone', 'broom'), repos = 'https://cran.ism.ac.jp')"