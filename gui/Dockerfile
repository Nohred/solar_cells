FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py audit.py metrics.py metrics_server.py triton_client.py ./

EXPOSE 8501 8502

CMD ["sh", "-c", "rm -rf /tmp/prometheus; mkdir -p /tmp/prometheus; python metrics_server.py & exec streamlit run app.py --server.address=0.0.0.0 --server.port=8501"]
