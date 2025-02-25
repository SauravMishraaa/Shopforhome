From python:3.10-slim
LABEL authors="Saurav"

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

CMD ["uvicorn","app:app", "--host","0.0.0.0", "--port","8000"]

