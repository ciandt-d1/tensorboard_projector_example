

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm

import argparse
import os

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorboard.plugins import projector

import models

import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def images_to_sprite(data):
    """Creates the sprite image along with any necessary padding
    Args:
        data: NxHxW[x3] tensor containing the images.
    Returns:
        data: Properly shaped HxWx3 image with any necessary padding.

    Source: https://github.com/tensorflow/tensorflow/issues/6322
    """
    if len(data.shape) == 3:
        data = np.tile(data[..., np.newaxis], (1, 1, 1, 3))
    data = data.astype(np.float32)
    min = np.min(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) - min).transpose(3, 0, 1, 2)
    max = np.max(data.reshape((data.shape[0], -1)), axis=1)
    data = (data.transpose(1, 2, 3, 0) / max).transpose(3, 0, 1, 2)
    # Inverting the colors seems to look better for MNIST
    data = 1 - data

    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = ((0, n ** 2 - data.shape[0]), (0, 0),
               (0, 0)) + ((0, 0),) * (data.ndim - 3)
    data = np.pad(data, padding, mode='constant',
                  constant_values=0)
    # Tile the individual thumbnails into an image.
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
                                                           + tuple(range(4, data.ndim + 1)))
    data = data.reshape(
        (n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = (data * 255).astype(np.uint8)
    return data


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--layer_name", required=True)

    args = parser.parse_args()

    # Get dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz')

    # Add channel (28,28) -> (28,28,1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Build and load trained model
    fine_tuned_model = models.get_mnist_model()
    fine_tuned_model.load_weights(filepath=args.ckpt_path)
    logger.info(fine_tuned_model.summary())

    # From a trained model, build another to act as a feature extractor
    feature_extractor = tf.keras.Model(inputs=fine_tuned_model.input,
                                       outputs=fine_tuned_model.get_layer(args.layer_name).output)

    projector_dir = os.path.join(args.output_dir, "tensorboard", "projector")
    os.makedirs(projector_dir, exist_ok=True)
    metadata_file = open(os.path.join(
        projector_dir, "metadata_classes.tsv"), "w")
    # metadata_file.write("Class\n") # If you have only one feature, you don't have to specify a header

    # Extract embeddings and Save label metadata
    images_list = []
    feature_vectors = []

    for image_label, image_np in tqdm(zip(y_test, x_test)):

        images_list.append(image_np)
        image_tensor_preproc = np.expand_dims(image_np, axis=0)
        image_embedding = np.squeeze(feature_extractor(
            image_tensor_preproc.astype(np.float32)))

        feature_vectors.append(image_embedding.tolist())
        metadata_file.write('{}\n'.format(image_label))
    metadata_file.close()

    feature_vectors = np.array(feature_vectors)
    images_arr = np.array(images_list)

    # Create sprite to be displayed at tensorboard
    sprite = images_to_sprite(images_arr)
    cv2.imwrite(os.path.join(projector_dir, 'sprite.png'), sprite)

    with tf.compat.v1.Session() as sess:

        # Save embeddings as a .ckpt file
        features = tf.Variable(feature_vectors, name='features')
        sess.run(features.initializer)
        saver = tf.compat.v1.train.Saver([features])
        saver.save(sess, os.path.join(projector_dir, 'features.ckpt'))

        # Create projector
        config = projector.ProjectorConfig()
        embedding = config.embeddings.add()
        embedding.tensor_name = features.name
        embedding.metadata_path = os.path.join(
            projector_dir, 'metadata_classes.tsv')

        embedding.sprite.image_path = os.path.join(projector_dir, 'sprite.png')
        embedding.sprite.single_image_dim.extend(
            [images_arr.shape[1], images_arr.shape[2], images_arr.shape[3]])
        projector.visualize_embeddings(
            tf.compat.v1.summary.FileWriter(projector_dir), config)
