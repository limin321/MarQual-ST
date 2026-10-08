# MarQual-ST image: the conda environment of environment.yml (the one tested by CI) + the package.
#   docker build -t marqual-st .
#   docker run --rm --user "$(id -u):$(id -g)" -v /path/to/project:/data \
#       marqual-st run -p /data/params.csv -m /data/marker_sets.csv
# Inside the container, paths in params.csv are container paths (/data/...) or relative to params.csv.
FROM mambaorg/micromamba:1.5.10-jammy

LABEL org.opencontainers.image.title="MarQual-ST" \
      org.opencontainers.image.description="Marker-based QC and niche discovery for binned spatial transcriptomics" \
      org.opencontainers.image.licenses="MIT"

ENV MPLBACKEND=Agg \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache \
    MPLCONFIGDIR=/tmp/matplotlib

# 1) environment (cached layer: only rebuilt when environment.yml changes)
COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml /tmp/env/
RUN micromamba install -y -n base -f /tmp/env/environment.yml && \
    micromamba clean --all --yes

# 2) the package
COPY --chown=$MAMBA_USER:$MAMBA_USER pyproject.toml README.md LICENSE /opt/marqual-st/
COPY --chown=$MAMBA_USER:$MAMBA_USER src /opt/marqual-st/src
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-cache-dir --no-deps /opt/marqual-st && \
    marqual-st --version && \
    rm -rf /tmp/numba_cache /tmp/matplotlib && \
    mkdir -m 1777 /tmp/numba_cache /tmp/matplotlib
# The check above imports scanpy and creates the numba / matplotlib caches as the build user. They are
# recreated empty and writable for every user (sticky bit, like /tmp): with `docker run --user ...`
# numba otherwise cannot write its cache and scanpy fails to import ("no locator available").

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "marqual-st"]
CMD ["--help"]
