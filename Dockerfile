FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CHECKPOINT_PATH=artifacts/models/strong_best.pt
ENV IMAGE_SIZE=224
ENV TOP_K=3

EXPOSE 8000

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
