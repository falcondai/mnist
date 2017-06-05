import importlib
import argparse
import tensorflow as tf

def build_parser():
    parser = argparse.ArgumentParser()
    # prefix arguments with `--m-`

    return parser


def build_model(images):
    net = images

    net = tf.contrib.layers.flatten(net)

    net = tf.contrib.layers.fully_connected(
        inputs=net,
        num_outputs=10,
        activation_fn=None,
        biases_initializer=tf.zeros_initializer(),
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        scope='fc1',
    )

    return net
