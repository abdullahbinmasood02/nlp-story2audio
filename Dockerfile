FROM python:3.9-slim

# Install dependencies including espeak-ng
RUN apt-get update && apt-get install -y \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Generate gRPC files
RUN python -m grpc_tools.protoc -I./api --python_out=./api --grpc_python_out=./api ./api/story_audio.proto

# Expose port for gRPC server
EXPOSE 50051

# Run the server
CMD ["python", "api/server.py"]