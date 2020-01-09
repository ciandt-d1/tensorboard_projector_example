FROM tensorflow/tensorflow:2.0.0-gpu-py3

RUN apt-get update && apt-get install -y

RUN apt install -y  libsm6 libxext6 libxrender-dev

WORKDIR /app

# RUN apt install wget \
#     && mkdir /assets \
#     && wget https://files.pythonhosted.org/packages/25/44/47f0722aea081697143fbcf5d2aa60d1aee4aaacb5869aee2b568974777b/tensorflow_gpu-2.0.0-cp36-cp36m-manylinux2010_x86_64.whl -O /assets/tensorflow_gpu-2.0.0-cp36-cp36m-manylinux2010_x86_64.whl

RUN mkdir /assets

COPY requirements.txt /assets/requirements.txt
RUN pip install -r /assets/requirements.txt --upgrade

COPY src /app/
