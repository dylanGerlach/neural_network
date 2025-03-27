import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf


def train_model(model, cifar_tr_inputs, cifar_tr_labels, batch_size, epochs):
    cifar_tr_inputs = cifar_tr_inputs.astype("float32") / 255.0
    print(model, cifar_tr_inputs.shape, cifar_tr_labels.shape, batch_size, epochs)
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(),
        optimizer="adam",
        metrics=["accuracy"],
    )

    model.fit(cifar_tr_inputs, cifar_tr_labels, epochs=epochs)
    return


def load_and_refine(filename, training_inputs, training_labels, batch_size, epochs):
    pretrained_model = keras.models.load_model(filename)
    num_layers = len(pretrained_model.layers)

    small_num_classes = len(np.unique(training_labels))

    refined_model = keras.Sequential(
        [keras.Input(shape=training_inputs.shape[1:])]
        + pretrained_model.layers[0 : num_layers - 1]
        + [layers.Dense(small_num_classes, activation="softmax")]
    )

    for i in range(0, num_layers - 1):
        refined_model.layers[i].trainable = False

    refined_model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    refined_model.fit(
        training_inputs, training_labels, batch_size=batch_size, epochs=epochs
    )

    return refined_model


def evaluate_my_model(model, test_inputs, test_labels):
    test_inputs = np.expand_dims(test_inputs, axis=-1)

    test_inputs = tf.image.resize(test_inputs, [32, 32])

    test_inputs = tf.image.grayscale_to_rgb(test_inputs)

    test_inputs = tf.cast(test_inputs, tf.float32)

    loss, accuracy = model.evaluate(test_inputs, test_labels, verbose=0)
    return accuracy
