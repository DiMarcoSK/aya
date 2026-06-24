FROM python:3.11-slim

WORKDIR /app

# Torch CPU-only wheel keeps the image small; the project is designed to
# run without a GPU (see README "System Requirements").
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]
