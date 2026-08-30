FROM python:3.10-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Crea carpetas que usa el codigo en tiempo de ejecucion
RUN mkdir -p /app/data /app/reports /app/models && chmod -R 777 /app

# HF Spaces usa el puerto 7860
 EXPOSE 10000 
 
 CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} 