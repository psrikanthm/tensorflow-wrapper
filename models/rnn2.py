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
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, config.num_steps, config.nr_features], name = 'X')
        self.y = tf.placeholder(dtype=tf.int32, shape=[None, config.num_steps], name = 'y')

	# this will be used to create the LSTMStateTuple. 
	# each layer has a state
	# each state consists of a hidden and a cell state
	# there is a hidden and cell state for each batch slice as it is fed through the model
	# and each of the nodes in hidden layers has both a cell state and a hidden state for each batch slice
        self.state = tf.placeholder(tf.float32, [config.num_layers, 2, None, config.hidden_size])
        # 2 is for cell_state and hidden_state

        self.opt = tf.train.AdamOptimizer()
        self.init = {} 
        self.saver = {}
        self.fetches = {}

    def define(self):
        """
        Declare all the variables and forward propagation here
        So that fetches can be setup in the end
        """
        W2 = tf.Variable(np.random.rand(self.config.hidden_size, self.config.class_size),\
                            name='weight_2', dtype=tf.float32)
        b2 = tf.Variable(np.random.rand(1,self.config.class_size), name='bias2', dtype=tf.float32)
        
        inputs_series = tf.split(self.X, self.config.num_steps, axis=1)
        inputs_series = [tf.squeeze(tensor, axis=1) for tensor in inputs_series]
        #inputs_series = tf.unstack(self.X, axis=1) 
        #inputs_series = self.X
        labels_series = tf.unstack(self.y, axis=1)


        # Since the TF Multilayer LSTM-API accepts state as a tuple of LSTMTuples,
        # we need to unpack self.state into this structure
        # For each layer in the self.state we create a LSTMTuple and put these 
        # in a tuple
        state_per_layer = tf.unstack(self.state, axis=0)
        rnn_tuple_state = tuple([
            tf.nn.rnn_cell.LSTMStateTuple(state_per_layer[i][0], state_per_layer[i][1]) \
            for i in range(self.config.num_layers)])

        # multiple rnn layers
        # MultiLayered LSTM is created by first making a single LSTMCell and 
        # then duplicating this cell in an array, supplying it to MultiRNNCell API
        self.first_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, self.config.nr_features,\
                    state_is_tuple=True)
        #self.first_cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, \
        #            state_is_tuple=True)
        cell = tf.nn.rnn_cell.LSTMCell(self.config.hidden_size, state_is_tuple=True)
        cell = tf.nn.rnn_cell.MultiRNNCell([self.first_cell] + ([cell] * (self.config.num_layers-1)), \
                state_is_tuple=True)

        # hidden -> hidden
        outputs, updated_state = tf.nn.static_rnn(cell, inputs_series, initial_state=rnn_tuple_state)
        #layer1_dim = first_cell.weights
        #layer1_dim = first_cell.trainable_variables
        #layer1_dim = first_cell.input_shape

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
                        #'debug':layer1_dim}

        return
