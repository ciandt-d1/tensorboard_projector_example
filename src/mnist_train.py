
import os
import datetime

import tensorflow as tf
from tensorflow.keras import backend as K
import numpy as np

import models

import argparse
import logging
logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", required=False, default=32, type=int)
    parser.add_argument("--epochs", required=False, default=5, type=int)

    args = parser.parse_args()

    # Download Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data(
        path='mnist.npz')

    # Add channel (28,28) -> (28,28,1)
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)

    # Normalize data
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Set output directory
    datetime_now_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = os.path.join(
        args.output_dir, datetime_now_str)
    ckpt_dir = os.path.join(output_dir, "model.hdf5")
    tensorboard_dir = os.path.join(output_dir, "tensorboard")

    # Create callbacks
    callback_save_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_dir,
                                                            monitor="val_loss",
                                                            save_best_only=True,
                                                            save_weights_only=True,
                                                            save_freq="epoch")

    callback_tensorboard = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_dir,
                                                          histogram_freq=0,
                                                          write_graph=True,
                                                          write_images=False,
                                                          profile_batch=0,
                                                          update_freq="epoch")

    # Build model
    model = models.get_mnist_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
                  metrics=["acc"],
                  loss="sparse_categorical_crossentropy")
    logger.info(model.summary())

    # Train and save
    model.fit(x=x_train, y=y_train,
              batch_size=args.batch_size,
              callbacks=[callback_save_ckpt, callback_tensorboard],
              epochs=args.epochs,
              shuffle=True,
              validation_data=(x_test, y_test)
              )
