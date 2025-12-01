FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    
WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt
    
COPY src/ ./src/
COPY notebooks/ ./notebooks/

EXPOSE 8500

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8500"]