# Use slim python
FROM python:3.11-slim

# Install system build dependencies required by many Python packages
# and packages that need to compile native extensions (numpy, scipy, etc.).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       git \
       curl \
       ca-certificates \
       libssl-dev \
       libffi-dev \
       python3-dev \
       pkg-config \
       gfortran \
       libopenblas-dev \
       liblapack-dev \
       cargo \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip / wheel / setuptools to ensure binary wheels are preferred
RUN pip install --upgrade pip setuptools wheel

# Install poetry via pip (use the image python to ensure installed in system site-packages)
RUN pip install poetry

# Prevent Poetry from creating virtualenvs inside the container (we want system site-packages)
ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_VIRTUALENVS_IN_PROJECT=false

WORKDIR /app

# Copy project files early so poetry can detect and install the project package
# We still copy only metadata first to leverage caching when dependencies are
# unchanged, but some projects require source present for editable installs.
COPY pyproject.toml poetry.lock* ./
COPY . .

# Run poetry install. Note: some very large packages (e.g., torch) may still fail
# to install if no compatible wheel is available for the slim image; consider
# moving heavy optional dependencies to extras or installing via pip wheels.
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Expose the API port used by docker-compose
EXPOSE 8000

# Default command: run the web API. Override in docker-compose for worker or other roles.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]