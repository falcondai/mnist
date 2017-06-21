import importlib
import argparse
import tensorflow as tf

def build_parser():
    parser = argparse.ArgumentParser()
    # prefix arguments with `--m-`
    parser.add_argument('--m-fc-width', dest='fc_width', type=int, default=128)
    parser.add_argument('--m-activation', dest='activation', default='relu')

    return parser


def build_model(images, fc_width, activation):
    activation_fn = getattr(tf.nn, activation)

    net = images

    net = tf.contrib.layers.flatten(net)

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=fc_width,
        activation_fn=activation_fn,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='fc1',
    )

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=10,
        activation_fn=None,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='fc2',
    )

    return net
