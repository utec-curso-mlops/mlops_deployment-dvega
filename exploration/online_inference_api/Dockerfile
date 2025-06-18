# Use official Python image
FROM python:3.8

# Set working directory
WORKDIR /app

# Copy the model into the container
COPY model /app/model

# Copy files
COPY app.py requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 8000

# Set entrypoint
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
