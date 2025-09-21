FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY code/deployment/app/requirements.txt .
RUN pip install --no-cache-dir --timeout=1000 --retries=5 -r requirements.txt

COPY code/deployment/app/ .

# Expose port
EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
