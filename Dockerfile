# Use an official Python runtime as a parent image.
FROM python:3.10-slim

# Set environment variables to prevent Python from writing .pyc files
# and to output logs directly.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies (if required). For example, gcc is needed to compile some packages.
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container.
COPY req.txt /app/

# Install Python dependencies.
RUN pip install --upgrade pip && pip install -r req.txt

# Copy the rest of the application code into the container.
COPY . /app/

# Expose a port if your application has a web component.
EXPOSE 8000

# Command to run the application. It assumes that your main application file is `main.py`
# and that your configuration file is `config.yaml`.
CMD ["python", "vul_app.py", "--config", "config.yaml"]
