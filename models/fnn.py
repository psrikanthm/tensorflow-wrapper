import tensorflow as tf
import numpy as np

from models.scg import SCG

class FNN(SCG):
    def __init__(self, config):
        """
        Initialize all the placeholders here
        so that feed_dict can be setup here
        """
        self.weight_initializer = tf.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=1)
        self.bias_initializer = tf.zeros_initializer()
    
        self.config=config
        self.X = tf.placeholder(dtype=tf.float32, shape=[None, config.num_features], name = 'X')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name = 'y')
        self.opt = tf.train.AdamOptimizer()

        self.init = {} 
        self.saver = {}

        self.fetches = {}

    def define(self):
        """
        Declare all the variables and forward propagation here
        So that fetches can be setup here
        """
        #Layer1: 
        W_hidden1 = tf.Variable(self.weight_initializer([self.config.num_features, \
                        self.config.hidden_layers[0]]), dtype=tf.float32)
        bias_hidden1 = tf.Variable(self.bias_initializer([self.config.hidden_layers[0]]), dtype=tf.float32)
        hidden1 = tf.nn.relu(tf.add(tf.matmul(self.X, W_hidden1), bias_hidden1))
        
        #Layer2: 
        W_hidden2 = tf.Variable(self.weight_initializer([self.config.hidden_layers[0], \
                    self.config.hidden_layers[1]]), dtype=tf.float32)
        bias_hidden2 = tf.Variable(self.bias_initializer([self.config.hidden_layers[1]]), dtype=tf.float32)
        hidden2 = tf.nn.relu(tf.add(tf.matmul(hidden1, W_hidden2), bias_hidden2))
        
        #Layer3: 
        W_hidden3 = tf.Variable(self.weight_initializer([self.config.hidden_layers[1],\
                    self.config.hidden_layers[2]]), dtype=tf.float32)
        bias_hidden3 = tf.Variable(self.bias_initializer([self.config.hidden_layers[2]]), dtype=tf.float32)
        hidden3 = tf.nn.relu(tf.add(tf.matmul(hidden2, W_hidden3), bias_hidden3))
        
        #Layer4: 
        W_hidden4 = tf.Variable(self.weight_initializer([self.config.hidden_layers[2], \
                    self.config.hidden_layers[3]]), dtype=tf.float32)
        bias_hidden4 = tf.Variable(self.bias_initializer([self.config.hidden_layers[3]]), dtype=tf.float32)
        hidden4 = tf.nn.relu(tf.add(tf.matmul(hidden3, W_hidden4), bias_hidden4))

        #output 
        W_out = tf.Variable(self.weight_initializer([self.config.hidden_layers[3], \
                    self.config.num_outputs]), dtype=tf.float32)
        bias_out = tf.Variable(self.bias_initializer([self.config.num_outputs]), dtype=tf.float32)
        out = tf.transpose(tf.add(tf.matmul(hidden4, W_out), bias_out))

        #cost
        error = tf.reduce_mean(tf.squared_difference(out, self.y))

        train_op = self.opt.minimize(error)

        self.saver = tf.train.Saver(max_to_keep=5)
        #it is mandatory to define global variables initializer since it is used to initialize
        # all the variables in computation graph
        self.init = tf.global_variables_initializer()
        # these are the operations that are passed in the sess.run() functin which
        # we would like to get evaluated
        self.fetches = {'error':error, 'train_op':train_op, 'output':out}
        return
