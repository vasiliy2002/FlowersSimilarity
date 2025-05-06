FROM python:3.11

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

COPY . /app

CMD ["python", "/app/main.py", "--host", "0.0.0.0" ]
