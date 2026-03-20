FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts and app
COPY models/ models/
COPY app.py .

EXPOSE 9696

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9696"]
