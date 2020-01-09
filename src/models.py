
import tensorflow as tf
from tensorflow.keras import backend as K
# tf.compat.v1.enable_eager_execution()


def get_mnist_model():
    """Creates a simple convolutional neuron network model
    
    Returns:
        [tf.keras.Model] -- Keras Model
    """ 
    input_layer = tf.keras.layers.Input(name="input_image", shape=(28, 28, 1))
    model = tf.keras.layers.Conv2D(32, kernel_size=(
        3, 3), activation=None, padding="same")(input_layer)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation(activation="relu")(model)

    model = tf.keras.layers.Conv2D(32, kernel_size=(
        3, 3), activation=None, padding="same")(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation(activation="relu")(model)

    model = tf.keras.layers.MaxPool2D((2, 2))(model)

    model = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), activation=None, padding="same")(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation(activation="relu")(model)

    model = tf.keras.layers.Conv2D(64, kernel_size=(
        3, 3), activation=None, padding="same")(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation(activation="relu")(model)

    model = tf.keras.layers.MaxPool2D((2, 2))(model)

    model = tf.keras.layers.Conv2D(128, kernel_size=(
        3, 3), activation=None, padding="same")(model)
    model = tf.keras.layers.BatchNormalization()(model)
    model = tf.keras.layers.Activation(activation="relu")(model)

    model = tf.keras.layers.GlobalMaxPooling2D()(model)

    for dense_units_i, dense_units in enumerate([64, 64]):
        model = tf.keras.layers.Dense(
            dense_units,
            activation=None,
            kernel_regularizer=tf.keras.regularizers.l2(
                l=5e-4),
            bias_regularizer=tf.keras.regularizers.l2(
                l=5e-4),
            name="model_dense_{}".format(dense_units_i))(model)
        model = tf.keras.layers.BatchNormalization(
            name="model_batch_norm_{}".format(dense_units_i))(model)
        model = tf.keras.layers.Activation(
            activation="relu", name="model_relu_{}".format(dense_units_i))(model)
        model = tf.keras.layers.Dropout(
            rate=0.5, name="model_dropout_{}".format(dense_units_i))(model)

    model = tf.keras.layers.Dense(
        10, activation='softmax', name='target')(model)

    model = tf.keras.Model(inputs=input_layer, outputs=model)

    return model
