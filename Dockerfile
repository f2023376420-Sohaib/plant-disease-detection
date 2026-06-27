# Base image
FROM python:3.11-slim

# Working directory
WORKDIR /app

# Requirements file copy karein
COPY requirements.txt .

# Packages install karein
RUN pip install --no-cache-dir -r requirements.txt

# Baqi code copy karein
COPY . .

# Port expose karein
EXPOSE 8000

# Command to run API
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]