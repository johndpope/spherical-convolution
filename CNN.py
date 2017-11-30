import argparse
import sys
sys.path.append("../deepfold/") # append path to deepfold library
import tempfile

import glob
import os
import time
import datetime
import math

import pickle
import numpy as np
import pandas as pd

from functools import reduce
import operator

from directional import conv_spherical_cubed_sphere, avg_pool_spherical_cubed_sphere
import tensorflow as tf

from Bio.PDB import PDBParser, MMCIFParser, Polypeptide
from Bio.PDB.Polypeptide import PPBuilder
from Bio.PDB import PDBList

FLAGS = None

# Set flags
flags = tf.app.flags

flags.DEFINE_string("input_dir", "./data/atomistic_features_cubed_sphere_train/", "Input path")
flags.DEFINE_float("test_set_fraction", 0.25,"Fraction of data set aside for testing")
flags.DEFINE_integer("validation_set_size", 10, "Size of validation set")
flags.DEFINE_string("logdir", "tmp/summary/", "Path to summary files")
flags.DEFINE_boolean("train", False, "Define if this is a training session")
flags.DEFINE_boolean("infer", False, "Define if this is a infering session")

FLAGS = flags.FLAGS

class CNNCubedSphereModel(object):
    """deepnn builds the graph for a deep net for classifying residues.

    Args:
    x: An input tensor with the dimensions (batch_size, sides, radius, xi, eta, channels).
    y: Amount of classes to predict
    shape: Shape of the input tensor x

    Returns:
    A tuple (y, keep_prob). y is a tensor of shape (batch_size, n_classes), with values
    equal to the logits of classifying the digit into one of n-classes (the
    digits 0-20). keep_prob is a scalar placeholder for the probability of
    dropout.
    """
    def __init__(self):
        tf.reset_default_graph()
        # internal setting
        self.optimizer = tf.train.AdamOptimizer(0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
        self.bias_initializer = tf.constant_initializer(0.0)
        self.shape = [-1, 6, 24, 38, 38, 2]
        self.n_classes = 21

        # placeholders
        self.labels = tf.placeholder(tf.float32, shape=[None, self.n_classes])
        self.x = tf.placeholder(tf.float32, shape=[None, 6, 24, 38, 38, 2])
        self.keep_prob = tf.placeholder_with_default(tf.constant(1.0), shape=None)

        # config
        self.batch_size = 10
        self.max_steps = 60000

        # build network
        self.graph = self._build_graph()

        # session and saver
        self.session = tf.Session()
        self._load_model()

        # initialize global variables
        tf.global_variables_initializer().run(session=self.session)
        print("Variables initialized")


    def _weight_variable(self, name, shape, stddev=0.1):
        return tf.get_variable(name, shape, initializer=tf.truncated_normal_initializer(stddev=stddev, dtype=tf.float32))

    def _reduce_dim(self, x):
            return reduce(operator.mul, x.shape.as_list()[1:], 1)

    def _fc_layer(self, input, channels_in, channels_out, name="fc"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.nn.relu(tf.matmul(input, W) + b)

    def _out_layer(self, input, channels_in, channels_out, name="out"):
        with tf.variable_scope(name):
            W = self._weight_variable("weights", [channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            return tf.matmul(input, W) + b

    def _conv_layer(self, input, channels_in, channels_out, name="conv"):
        with tf.variable_scope(name) as scope:
            W = self._weight_variable("weights", [3, 3, 3, channels_in, channels_out])
            b = tf.get_variable("b", shape=[channels_out], initializer=self.bias_initializer, dtype=tf.float32)
            convolution = conv_spherical_cubed_sphere(input, W, strides=[1, 1, 1, 1, 1], padding="SAME", name=name)
            activation = tf.nn.relu(convolution + b)
            return avg_pool_spherical_cubed_sphere(activation, ksize=[1, 1, 3, 3, 1], strides=[1,1,2,2,1], padding="VALID")

    def _build_graph(self):
        # Reshape to use within a convolutional neural net.
        x = tf.reshape(self.x, shape=[-1, 6, 24, 38, 38, 2])
        conv1 = self._conv_layer(x, 2, 16,  name="conv1")
        conv2 = self._conv_layer(conv1, 16, 32,  name="conv2")
        conv3 = self._conv_layer(conv2, 32, 64,  name="conv3")
        conv4 = self._conv_layer(conv3, 64, 128,  name="conv4")

        flattened = tf.reshape(conv4,  [-1, self._reduce_dim(conv4)])

        fc1 = self._fc_layer(flattened, self._reduce_dim(conv4), 2048, name="fc1")
        drop_out = tf.nn.dropout(fc1, self.keep_prob)
        fc2 = self._fc_layer(drop_out, 2048, 2048, name="fc2")
        return self._out_layer(fc2, 2048, self.n_classes)

    def _batch_factory(self):
        # get proteins feature file names and grid feature file names
        protein_feature_filenames = sorted(
            glob.glob(os.path.join(FLAGS.input_dir, "*protein_features.npz")))
        grid_feature_filenames = sorted(
            glob.glob(os.path.join(FLAGS.input_dir, "*residue_features.npz")))

        # Set range for validation and test set
        validation_end = test_start = int(
            len(protein_feature_filenames) * (1. - FLAGS.test_set_fraction))
        train_end = validation_start = int(
            validation_end - FLAGS.validation_set_size)

        # create object from BatchFactory class
        train_batch_factory = BatchFactory()

        # add the dataset X labels
        train_batch_factory.add_data_set("data",
                                        protein_feature_filenames[:train_end],
                                        grid_feature_filenames[:train_end])

        # add the dataset Y labels
        train_batch_factory.add_data_set("model_output",
                                        protein_feature_filenames[:train_end],
                                        key_filter=["aa_one_hot"])

        # create object from BatchFactory class
        validation_batch_factory = BatchFactory()

        # add the dataset X labels
        validation_batch_factory.add_data_set("data",
                                            protein_feature_filenames[
                                                validation_start:validation_end],
                                            grid_feature_filenames[validation_start:validation_end])

        # add the dataset Y labels
        validation_batch_factory.add_data_set("model_output",
                                            protein_feature_filenames[
                                                validation_start:validation_end],
                                            key_filter=["aa_one_hot"])

        # create object from BatchFactory class
        test_batch_factory = BatchFactory()

        # add the dataset X labels
        test_batch_factory.add_data_set("data",
                                        protein_feature_filenames[test_start:],
                                        grid_feature_filenames[test_start:])

        # add the dataset Y labels
        test_batch_factory.add_data_set("model_output",
                                        protein_feature_filenames[test_start:],
                                        key_filter=["aa_one_hot"])

        return {'train': train_batch_factory, 'validation': validation_batch_factory, 'test': test_batch_factory}

    def _loss(self, logits, labels):
        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
            return tf.reduce_mean(cross_entropy)

    def _accuracy(self, logits, labels):
        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(self.labels, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            return tf.reduce_mean(correct_prediction)

    def _probabilities(self, logits):
        with tf.name_scope('probabilities'):
            return tf.nn.softmax(logits=logits)

    def _load_model(self):
        # Load model
        saver = tf.train.Saver()
        ckpt = tf.train.get_checkpoint_state('model/')
        saver.restore(self.session, ckpt.model_checkpoint_path)

    def infer(self, data):
        return self.session.run([self._probabilities(self.graph)], feed_dict={self.x: data})

def main(_):
    print("hej")

if __name__ == '__main__':
    tf.app.run()
