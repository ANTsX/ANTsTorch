FROM antsx/antstorch:latest

USER root
WORKDIR /home/antstorchuser

RUN . "${VIRTUAL_ENV}/bin/activate" && \
    /opt/bin/download_antstorch_data.py \
        --cache-dir /home/antstorchuser/.antstorch \
        --strict && \
    chmod -R 0755 /home/antstorchuser/.antstorch

USER antstorchuser

LABEL maintainer="Philip A Cook (https://github.com/cookpa)" \
      description="ANTsTorch is part of the ANTsX ecosystem (https://github.com/ANTsX). \
ANTsX citation: https://pubmed.ncbi.nlm.nih.gov/33907199"
ENTRYPOINT ["python"]
