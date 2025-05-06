# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy app and model files
COPY . /app
COPY models /app/models
# Dockerfile
COPY models/all-MiniLM-L6-v2 /app/models/all-MiniLM-L6-v2


# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

EXPOSE 8080

CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]
