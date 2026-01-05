# Use official Miniconda3 Linux image
FROM continuumio/miniconda3:latest

# WORKDIR already set
WORKDIR /app

# Copy everything from current context (server/) into /app
COPY . /app

# Copy environment file (already inside context)
COPY se3d_PYPC.yml /app/

# Install environment
RUN conda init bash && \
    conda env create -f se3d_PYPC.yml && \
    conda clean -afy

# Make RUN commands use the environment
SHELL ["conda", "run", "-n", "se3d", "/bin/bash", "-c"]

# Install any pip-only packages if needed (optional)
# RUN pip install some-package

# Expose the port Render uses
EXPOSE 10000

# Start command
CMD ["conda", "run", "-n", "se3d", "python", "src/index.py"]