FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
