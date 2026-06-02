# Sleepy Durian World Backend Dockerfile
# python 3.9 + InsightFace + API

FROM python:3.11-slim
LABEL authors="sleepydurian"

# System package for OpenCV and PostgreSQL
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# python dependencies (first copy then use Docker-Cache)
COPY requirements.txt .
RUN pip install --no-cache-dir --update pip && \
    pip install --no-cache-dir -r requirements.txt

# copy source code
  COPY . .

# Folder für datas
RUN mkdir -p /FacialRecognizationForAttendance/data/employees_photos /FacialRecognizationForAttendance/data/snapshots

EXPOSE 8000

#Uvicorn starts FastAPI
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--log-level", "info"]