FROM python:3.10-slim

COPY requirements.txt .

RUN apt update
RUN apt install git -y
RUN pip install --upgrade pip --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN apt install graphviz -y

COPY src ./src
RUN pip install -e src