# Tensorboard Embedding Projection Example
How to make embeddings projection on TensorBoard

![TensorBoard Projector](./images/TensorBoard.gif)

## Install docker

[Follow these steps](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

## Install nvidia-container-toolkit or nvidia-docker2

[Follow these steps](https://github.com/NVIDIA/nvidia-docker)

## Install and run

The command below will build the docker image and run other installation steps

``` 
make install
```

## Build docker image

This is already performed if you previously have run `make install` 

``` 
docker build . -t mnist_projection
```

Enter container
``` 
docker run -it \
-v ${PWD}/projections:/projections/ \
-v ${PWD}/keras_datasets:/root/.keras/datasets \
-p 6006:6006 \
--rm --gpus all mnist_projection bash
```

### Train model

```
python -m mnist_train \
    --output_dir /projections\
    --batch_size 16 \
    --epochs 5
```

### Extract and Visualize Embeddings

```
python -m mnist_project_embeddings \
    --output_dir /projections/<timestamp>/ \
    --ckpt_path /projections/<timestamp>/model.ckpt \
    --layer_name model_dense_1
```


### Visualize embeddings

```
tensorboard --logdir /projections/<timestamp>/tensorboard/projector/ --port 6006
```

Enter `localhost:6006` at your browser