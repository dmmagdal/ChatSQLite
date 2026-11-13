FROM mambaorg/micromamba:2.10.0

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://host.docker.internal:11434

COPY --chown=$MAMBA_USER:$MAMBA_USER env-docker.yml /tmp/env-docker.yml
RUN micromamba install --yes --name base --file /tmp/env-docker.yml \
    && micromamba clean --all --yes \
    && rm /tmp/env-docker.yml

COPY workflow.py helper.py /app/
WORKDIR /data

ENTRYPOINT ["/usr/local/bin/_entrypoint.sh", "python", "/app/workflow.py"]
