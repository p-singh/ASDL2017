from asyncio.windows_events import _BaseWaitHandleFuture

import tflearn
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib.legacy_seq2seq.python.ops import seq2seq
import numpy
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib import rnn
from collections import defaultdict

class Code_Completion_Baseline:

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        splitted = string.split("-@@-")
        return {"type": splitted[0], "value": splitted[1]}

    def one_hot(self, string):
        vector = [0] * len(self.string_to_number)
        vector[self.string_to_number[string]] = 1
        return vector

    def prepare_data(self, token_lists):
        # encode tokens into one-hot vectors
        all_token_strings = set()
        token_frequencies = defaultdict(int)
        all_tokens = 0
        for token_list in token_lists:
            for token in token_list:
                all_tokens += 1
                token_frequencies[self.token_to_string(token)] += 1
                all_token_strings.add(self.token_to_string(token))
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()

        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1
        print("TOTAL TOKEN OCCURRENCES: ", all_tokens)
        for t in token_frequencies:
            print(t, " :: ", token_frequencies[t]*100/all_tokens, " -- ", token_frequencies[t])

        # prepare x,y pairs
        xs = []
        ys = []

        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.string_to_number[previous_token_string])
                    ys.append(self.one_hot(token_string))

        print("x,y pairs: " + str(len(xs)))        
        return (xs, ys)

    def getIOarrays(self, token_lists):
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                if idx > 0:
                    token_string = self.token_to_string(token)
                    previous_token_string = self.token_to_string(token_list[idx - 1])
                    xs.append(self.one_hot(previous_token_string))
                    ys.append(self.one_hot(token_string))


    def create_network(self):
        self.seq2seq_model == "embedding_attention"
        mode = "train"
        GO_VALUE = self.out_max_int + 1
        self.net = tflearn.input_data(shape=[None, self.in_seq_len + self.out_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(self.net, [0, 0], [-1, self.in_seq_len], name="enc_in")  # get encoder inputs
        encoder_inputs = tf.unstack(encoder_inputs,
                                    axis=1)  # transform into list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(self.net, [0, self.in_seq_len], [-1, self.out_seq_len],
                                  name="dec_in")  # get decoder inputs
        decoder_inputs = tf.unstack(decoder_inputs,
                                    axis=1)  # transform into list of self.out_seq_len elements, each [-1]

        go_input = tf.multiply(tf.ones_like(decoder_inputs[0], dtype=tf.int32),
                               GO_VALUE)  # insert "GO" symbol as the first decoder input; drop the last decoder input
        decoder_inputs = [go_input] + decoder_inputs[
                                      : self.out_seq_len - 1]  # insert GO as first; drop last decoder input

        feed_previous = not (mode == "train")

        # if self.verbose > 3:
        #     print("feed_previous = %s" % str(feed_previous))
        #     print("encoder inputs: %s" % str(encoder_inputs))
        #     print("decoder inputs: %s" % str(decoder_inputs))
        #     print("len decoder inputs: %s" % len(decoder_inputs))

        self.n_input_symbols = self.in_max_int + 1  # default is integers from 0 to 9
        self.n_output_symbols = self.out_max_int + 2  # extra "GO" symbol for decoder inputs
        num_layers = 2
        single_cell = rnn.BasicLSTMCell(32)  #getattr(rnn_cell, cell_type)(cell_size, state_is_tuple=True)
        self.inSeqLen = 10
        self.outSeqLen = 10
        if num_layers == 1:
            cell = single_cell
        else:
            cell = rnn.MultiRNNCell([single_cell] * num_layers)

        if self.seq2seq_model == "embedding_rnn":
            model_outputs, states = seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                  # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                  decoder_inputs,
                                                                  cell,
                                                                  num_encoder_symbols=self.n_input_symbols,
                                                                  num_decoder_symbols=self.n_output_symbols,
                                                                  embedding_size=200,
                                                                  feed_previous=feed_previous)
        elif self.seq2seq_model == "embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                        # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                        decoder_inputs,
                                                                        cell,
                                                                        num_encoder_symbols=self.n_input_symbols,
                                                                        num_decoder_symbols=self.n_output_symbols,
                                                                        embedding_size=200,
                                                                        num_heads=1,
                                                                        initial_state_attention=False,
                                                                        feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model",
                             model_outputs)  # for TFLearn to know what to save and restore

        # model_outputs: list of the same length as decoder_inputs of 2D Tensors with shape [batch_size x output_size] containing the generated outputs.
        if self.verbose > 2: print("model outputs: %s" % model_outputs)
        network = tf.stack(model_outputs,
                           axis=1)  # shape [-1, n_decoder_inputs (= self.out_seq_len), num_decoder_symbols]
        if self.verbose > 2: print("packed model outputs: %s" % network)

        if self.verbose > 3:
            all_vars = tf.get_collection(tf.GraphKeys.VARIABLES)
            print("all_vars = %s" % all_vars)

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        self.net = tflearn.regression(network,
                                     placeholder=targetY,
                                     optimizer='adam',
                                     learning_rate=0.0001,
                                     loss=self.sequence_loss,
                                     metric=self.accuracy,
                                     name="Y")

        self.model = tflearn.DNN(self.net)

        # self.net = tflearn.input_data(shape=[None, len(self.string_to_number)])
        # self.net = tflearn.embedding(self.net, input_dim=100, output_dim=200)
        # self.net = tflearn.simple_rnn(self.net, 32)
        # self.net = tflearn.fully_connected(self.net, len(self.string_to_number), activation='softmax')
        # self.net = tflearn.regression(self.net)
        # self.model = tflearn.DNN(self.net)
    
    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        print(self.string_to_number)
        self.create_network()
        self.model.load(model_file)
    
    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        xs = xs[:len(xs)-(len(xs) % 5)]
        xs = numpy.reshape(xs, (-1,5))
        self.model.fit(xs, ys, n_epoch=1, batch_size=1024, show_metric=True)
        self.model.save(model_file)
        self.model.load(model_file)
        
    def query(self, prefix, suffix):
        previous_token_string = self.token_to_string(prefix[-1])
        x = self.one_hot(previous_token_string)
        y = self.model.predict([x])
        predicted_seq = y[0]
        if type(predicted_seq) is numpy.ndarray:
            predicted_seq = predicted_seq.tolist() 
        best_number = predicted_seq.index(max(predicted_seq))
        best_string = self.number_to_string[best_number]
        print(best_string)
        best_token = self.string_to_token(best_string)
        return [best_token]
    
