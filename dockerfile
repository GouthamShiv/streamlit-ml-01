FROM python:3.9-slim-buster

WORKDIR /app

COPY ["./app.py", "requirements.txt", "/app/"]

RUN mv /app/app.py /app/ML_classifiers_comparison.py \
    && pip3 install --no-cache-dir -r requirements.txt

CMD streamlit run ML_classifiers_comparison.py --server.enableCORS=false --server.enableXsrfProtection=false

EXPOSE 8501