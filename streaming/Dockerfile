FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY streaming/ ./streaming/
COPY model/ ./model/
COPY configs/ ./configs/
COPY data_generator/ ./data_generator/

RUN mkdir -p logs

# No CMD — docker-compose chooses producer or consumer per service
