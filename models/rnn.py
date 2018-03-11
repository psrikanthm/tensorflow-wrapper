from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np

import collections
from models.scg import SCG

class RNN(SCG):
    def __init__(self, config):
        """
        Initialize all the placeholders here
        so that feed_dict can be setup here
        """
        self.config=config
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, config.num_steps], name = 'X')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, config.num_steps], name = 'y')
        cell_state = tf.placeholder(tf.float32, [None, self.config.hidden_size], name='cell_state')
        hidden_state = tf.placeholder(tf.float32, [None, self.config.hidden_size], name='hidden_state')
        self.state = (cell_state, hidden_state)
        self.opt = tf.train.AdamOptimizer()

        self.init = {} 
        self.saver = {}

        self.fetches = {}

    def define(self):
        """
        Declare all the variables and forward propagation here
        So that fetches can be setup here
        """
        W2 = tf.Variable(np.random.rand(self.config.hidden_size, self.config.class_size),\
                            name='weight_2', dtype=tf.float32)
        b2 = tf.Variable(np.random.rand(1,self.config.class_size), name='bias2', dtype=tf.float32)
        
        inputs_series = tf.split(self.X, self.config.num_steps, axis=1)
        #inputs_series = tf.unstack(self.X, axis=1) 
        labels_series = tf.unstack(self.y, axis=1)

        #here we are passing the input data directly to rnn
        #layers. But we might want to pass it through other
        # layers before rnn
        self.__cell = tf.nn.rnn_cell.BasicLSTMCell(self.config.hidden_size, state_is_tuple=True)

        # input -> hidden
        outputs, updated_state = tf.nn.static_rnn(self.__cell, inputs_series, \
                    initial_state=self.state)

        #hidden-> output
        logits_series = [tf.matmul(output, W2) + b2 for output in outputs]
	predictions = [tf.nn.softmax(logits) for logits in logits_series]
       
        # error calculation using softmax since it is a classification problem 
        errors_series = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=labels)\
                        for logits,labels in zip(logits_series, labels_series)] 
        error = tf.reduce_mean(errors_series)
        
        #optiimization
        train_op = self.opt.minimize(error)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=5)

        self.fetches = {'error':error, 'train_op':train_op, 'state':updated_state, 'output':predictions}

        return
