# syntax=docker/dockerfile:1

FROM python:3.8.13-slim-buster

WORKDIR /app

RUN apt-get update && \
    apt-get -y install gcc mono-mcs && \
    rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip
RUN pip3 install streamlit-cropper streamlit-option-menu
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
# UN pip3 install streamlitR
# RUN

# Copy all & Run
COPY . .

EXPOSE 8501

CMD [ "streamlit", "run", "main.py" ]