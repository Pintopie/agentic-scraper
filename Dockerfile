# Playwright's official Python image — Chromium and all system deps are pre-installed.
# https://playwright.dev/python/docs/docker
FROM mcr.microsoft.com/playwright/python:v1.44.0-jammy

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py", "--config", "config.yaml"]
