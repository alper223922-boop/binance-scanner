FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY scanner.py .
# ⚠️ .env dosyası imajın içine gömülmemeli! docker-compose zaten mount ediyor.
CMD ["python", "-u", "scanner.py"]