FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn8-devel

RUN apt-get update && apt-get install -y git vim wget

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

ENV PYTHONPATH=/app
CMD ["/bin/bash"]
