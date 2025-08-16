FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for numpy/scipy/sklearn/cvxpy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc gfortran libatlas-base-dev libblas-dev liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

EXPOSE 8501

