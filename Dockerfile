FROM python:3.9-slim
WORKDIR /app

# Установка зависимостей
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Копируем исходники
COPY . .

# Открываем порт
EXPOSE 8001

# Запуск Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]