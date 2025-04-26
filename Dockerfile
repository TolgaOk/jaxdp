FROM tolgaok/cuda12.6.3-ub24-conda:latest

LABEL maintainer="Tolga Ok"
LABEL version="1.0"
LABEL description="Development environment for jaxdp"

# Re-declare ARGs inherited from the base image
ARG USERNAME=developer
ARG UID=1000
ARG GID=1000

ENV USERNAME=${USERNAME}

# Switch to root to install dependencies
USER root

# Copy the requirements file and install dependencies
COPY requirements.txt /tmp/requirements.txt
RUN chown $USERNAME:$USERNAME /tmp/requirements.txt && \
    conda run -n main pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Switch back to non-root user
USER ${USERNAME}
WORKDIR /home/${USERNAME}

EXPOSE 8080
CMD ["xvfb-run-safe", "code", "tunnel", "--accept-server-license-terms"]

