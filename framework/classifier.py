from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import tensorflow as tf
import numpy as np
import json

class Config:
    '''
    Config class to store the configuration from the json file
    '''
    def __init__(self, **entries):
        self.__dict__.update(entries)

class Classifier:
    '''
    It is a class emulating the behaviour of Scikit-Learn's regressor or classifier providing
    interface for the functions fit, validate, predict, score etc
    '''
    def __init__(self, config_file='config.json'):
        '''
        config_file = config json file
        '''
        self.config = Config(**json.load(open(config_file)))

    def initialize(self, Class):
        '''
        Initialize the model and construct the computation graph
        Class = Model Class which is a child of models.scg.SCG base class
        '''
        self.model = Class(self.config)
        self.model.define()

    def process_data(self, data_file, input_fn):
        '''
        process the data in such a way that self.model supports
        data_file = data file that is to be read
        input_fn = function to process the data in the data_file
        '''
        #load the data from the file based on different file formats
        # currently only .npy is supported, in future would like to support csv...
        if data_file.endswith('.npy'):
            data = np.load("{}/{}".format(self.config.datadir, data_file))
        data = data.astype('float32')

        Xdata, ydata = input_fn(data, self.config)

        self.config.class_size = int(np.max(ydata)) + 1

        #split the data into train and test
        t_len = len(Xdata)

        #split the data into train and test in the ration of train_test_split
        itrain = int(np.floor(self.config.train_test_split * t_len))
        Xtrain = Xdata[:itrain]
        ytrain = ydata[:itrain]
        Xtest = Xdata[itrain + 1:]
        ytest = ydata[itrain + 1:]

        return Xtrain, ytrain, Xtest, ytest
    
    def fit(self, X, y, save_model=False):
        """
        X = input training data
        y = output training data
        save_model = if True, save the model in self.config.logdir every epoch and close the session
                    else, close the session
        """

        # open the session and initialize the variable by calling model's initializer
        self.sess = tf.Session()
        self.sess.run(self.model.init)

        # if the state is to be included in the keys (for time series models)
        if self.config.fetches_with_state:
            keys = ['error','state','train_op']
        else:
            keys = ['error','train_op']
        fetches={k:self.model.fetches[k] for k in keys \
                if k in self.model.fetches}

        # for monitoring - tensorboard
        writer = tf.summary.FileWriter(self.config.logdir, self.sess.graph)

        # Make use of new Dataset to create universal pipelines for all types of data
        # For now only using for making batches by using the get_next iterator
        # This is very effective tool though, kb article - 
        #https://www.tensorflow.org/programmers_guide/datasets
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        dataset = dataset.batch(self.config.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        next_batch = iterator.get_next()
        iterator_init = iterator.make_initializer(dataset)

        for e in range(self.config.epochs):
            self.sess.run(iterator_init) #after every epoch run the initializer so that iterator points
                                        # to 1st element
            error = 0
            count = 0
            if self.config.fetches_with_state:
                # 2 is for cell state and hidden state
                # On each epoch start the state from all zeros
                _fetches = {'state': np.zeros((self.config.num_layers, 2, self.config.batch_size, \
                            self.config.hidden_size))}
            while True:
                feed_dict={}
                try:
                    feed_dict[self.model.X], feed_dict[self.model.y] = self.sess.run(next_batch)
                except tf.errors.OutOfRangeError:
                    print("debugc:out of range error")
                    break
                if feed_dict[self.model.X].shape[0] != self.config.batch_size: 
                    #sometimes it picks < batch_size elements, it is to avoid that
                    break

                if self.config.fetches_with_state:
                    feed_dict[self.model.state] = _fetches['state']
            
                _fetches = self.sess.run(fetches, feed_dict=feed_dict)
                error += _fetches['error']
                count += 1

            print("epoch={}, error={}, nr_batches={}".format(e,error/count, count))
           
            if save_model: #store the model only if session is getting closed
                self.model.saver.save(self.sess, "{}/{}".format(self.config.logdir, self.config.model_name), \
                            global_step=e)

        writer.close()
        
        if save_model: #if model is getting saved, this session is not needed anymore
            self.sess.close()

    def validate(self, X, y, reuse_model=False):
        """
        if reuse_model=False, the self.sess is not None and it contains the state of Graph.
        else, restore the session from the saved model from the self.config.checkpoint_file
        """
        self.config.batch_size = len(y)

        if reuse_model: #restore from checkpoint file
            self.sess = tf.Session()
            saver=tf.train.import_meta_graph("{}/{}".format(self.config.logdir, self.config.checkpoint_file))
            saver.restore(self.sess,tf.train.latest_checkpoint("{}/".format(self.config.logdir)))
        
        if self.config.fetches_with_state:
            keys = ['error', 'state']
            #at each epoch start the state from all zeros
            _fetches = {'state': np.zeros((self.config.num_layers, 2, self.config.batch_size, \
                            self.config.hidden_size))}
        else:
            keys = ['error']

        fetches={k:self.model.fetches[k] for k in keys if k in self.model.fetches}
        dataset = tf.data.Dataset.from_tensor_slices((X,y))
        dataset = dataset.batch(self.config.batch_size)
        iterator = tf.data.Iterator.from_structure(dataset.output_types)
        next_batch = iterator.get_next()
        iterator_init = iterator.make_initializer(dataset)
        self.sess.run(iterator_init)

        count = 0
        while True: 
            feed_dict={}
            try:
                feed_dict[self.model.X], feed_dict[self.model.y] = self.sess.run(next_batch)
            except tf.errors.OutOfRangeError:
                print("debugc:out of range error")
                break
            if feed_dict[self.model.X].shape[0] != self.config.batch_size:
                break

            if self.config.fetches_with_state:
                feed_dict[self.model.state] = _fetches['state']

            _fetches = self.sess.run(fetches, feed_dict=feed_dict)
            count += 1

            print("batch={}, error={}".format(count, _fetches['error']))

    def score(self, X, y):
        pass

    def __load_model(self):
        pass

    def __save_model(self):
        pass
