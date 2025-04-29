# Use the NVIDIA CUDA base image
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# Set environment variables to non-interactive
ENV DEBIAN_FRONTEND=noninteractive
# Update and install desired utilities
RUN apt-get update && apt-get install -y \
    vim \
    nano \
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m pip install --upgrade pip
#RUN pip install torch transformers datasets peft accelerate
#RUN mkdir /var/run/sshd
#RUN echo 'root:root' | chpasswd && \
#    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
## Expose SSH port (if needed)
EXPOSE 22
# Optionally, set an entrypoint or command
CMD [ "sh", "-c", "echo 'I'\''m a new pod with GPUs, storage, and utilities' && sleep infinity" ]