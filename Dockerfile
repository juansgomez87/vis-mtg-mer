FROM python:3.8.10

LABEL maintainer="juansebastian.gomez@upf.edu"

RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.2/dumb-init_1.2.2_amd64 \
    && chmod +x /usr/local/bin/dumb-init

# Common requirements
RUN apt-get update \
    && apt-get install -y \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir /code
WORKDIR /code

RUN pip3 install --upgrade pip
COPY ./requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache -r /tmp/requirements.txt

COPY . /code

EXPOSE 8050

CMD ["python3", "/code/app.py"]
