# syntax=docker/dockerfile:1

FROM python:3.11-slim
WORKDIR /bank_customer_churn_prediction
COPY . .
RUN pip3 install -r requirements.txt
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "server:app"]