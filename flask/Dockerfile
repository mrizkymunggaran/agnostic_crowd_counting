# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Copy all & Run
COPY . .

EXPOSE 8081

CMD [ "python3", "app.py" ]