# Use a slim, official Python image
FROM --platform=linux/amd64 python:3.11-slim

# Install system dependencies needed for downloading models
RUN apt-get update && apt-get install -y git git-lfs wget && git lfs install

# Set the working directory inside the container
WORKDIR /app

# Libs required by LightGBM/Scikit-learn
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

RUN mkdir -p llm_model semantic_model

RUN git clone https://huggingface.co/BAAI/bge-base-en-v1.5 semantic_model

RUN wget -O llm_model/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf

RUN wget -O semantic_model/pytorch_model.bin https://huggingface.co/BAAI/bge-base-en-v1.5/blob/main/pytorch_model.bin

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .

# Install all Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all your project files and data into the container
# This includes main.py, models/, llm_model/, main/, etc.
COPY . .

# Define the command that will be executed when the container starts
# It runs your main script and tells it to process the 'main' directory
CMD ["python", "main.py"]