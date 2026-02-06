FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir opencv-python-headless

COPY . .

RUN mkdir -p data models outputs uploads cache

ENV PYTHONPATH=/app
ENV DISPLAY=:99

EXPOSE 5000

CMD ["python", "web_interface.py", "--host", "0.0.0.0", "--port", "5000"]
