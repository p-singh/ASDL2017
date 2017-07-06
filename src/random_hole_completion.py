import tflearn
import numpy
import os
import io
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib import rnn
from collections import defaultdict



class Random_Hole_Completion:

    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        if (string == "HOLE") or (string == "ZERO") or string == "UNK":
            return {"type": "", "value": ""}
        else:
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

        all_token_strings.add("UNK")
        all_token_strings.add("ZERO")
        all_token_strings.add("HOLE")
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        print("Unique tokens: " + str(len(all_token_strings)))
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0

        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        print("TOTAL TOKEN OCCURRENCES: ", all_tokens)

        sorted_freq = [(k, token_frequencies[k]) for k in
                       sorted(token_frequencies, key=token_frequencies.get, reverse=True)]

        for k, v in sorted_freq:
            print(k, " :: ", v * 100 / all_tokens, " -- ", v)

        # prepare x,y pairs
        xs = []
        ys = []
        self.window = 10

        self.in_seq_len = self.window * 2 + 1
        self.out_seq_len = 5
        self.in_max_int = self.out_max_int = len(self.string_to_number)

        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                token_list[idx] = self.string_to_number[self.token_to_string(token)]

        for token_list in token_lists:
            token_list = token_list + [self.string_to_number["ZERO"]] * self.window
            for idx, token in enumerate(token_list):
                if idx <= len(token_list) - self.in_seq_len:
                    xs.append(token_list[idx:idx + self.window] + [self.string_to_number["HOLE"]] + token_list[idx + self.window + 1:idx + self.window + self.window + 1])
                    ys.append(token_list[idx + self.window:idx + self.window + 2] + [self.string_to_number["UNK"]]*3)

        for token_list in token_lists:
            token_list = token_list + [self.string_to_number["ZERO"]] * self.window
            for idx, token in enumerate(token_list):
                if idx <= len(token_list) - self.in_seq_len - 1:
                    xs.append(token_list[idx:idx + self.window] + [self.string_to_number["HOLE"]] + token_list[idx + self.window + 2:idx + self.window + self.window + 2])
                    ys.append(token_list[idx + self.window:idx + self.window + 3] + [self.string_to_number["UNK"]] * 2)

        for token_list in token_lists:
            token_list = token_list + [self.string_to_number["ZERO"]] * self.window
            for idx, token in enumerate(token_list):
                if idx <= len(token_list) - self.in_seq_len - 2:
                    xs.append(token_list[idx:idx + self.window] + [self.string_to_number["HOLE"]] + token_list[idx + self.window + 3:idx + self.window + self.window + 3])
                    ys.append(token_list[idx + self.window:idx + self.window + 4] + [self.string_to_number["UNK"]] * 1)
        print("x,y pairs: ", str(len(xs)), str(len(ys)))
        # self.write_parallel_text(xs, ys, "network_inputs/")
        xs = numpy.array(xs)
        ys = numpy.array(ys)
        return (xs, ys)

    def getIOarrays(self, token_lists):
        self.in_seq_len = 5
        self.out_seq_len = 5
        self.in_max_int = self.out_max_int = len(self.string_to_number)
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
        self.seq2seq_model = "embedding_attention"
        mode = "train"
        GO_VALUE = self.out_max_int + 1

        self.net = tflearn.input_data(shape=[None, self.in_seq_len], dtype=tf.int32, name="XY")
        encoder_inputs = tf.slice(self.net, [0, 0], [-1, self.in_seq_len], name="enc_in")  # get encoder inputs
        encoder_inputs = tf.unstack(encoder_inputs, axis=1)  # transform to list of self.in_seq_len elements, each [-1]

        decoder_inputs = tf.slice(self.net, [0, 0], [-1, self.out_seq_len], name="dec_in")
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)  # transform into list of self.out_seq_len elements

        go_input = tf.multiply(tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE)
        decoder_inputs = [go_input] + decoder_inputs[
                                      : self.out_seq_len - 1]  # insert GO as first; drop last decoder input

        feed_previous = not (mode == "train")

        self.n_input_symbols = self.in_max_int + 1  # default is integers from 0 to 9
        self.n_output_symbols = self.out_max_int + 2  # extra "GO" symbol for decoder inputs

        cells = []
        for _ in range(3):
            cells.append(self.getCell(128))

        cell = rnn.MultiRNNCell(cells)

        if self.seq2seq_model == "embedding_rnn":
            model_outputs, states = seq2seq.embedding_rnn_seq2seq(encoder_inputs,
                                                                  # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                  decoder_inputs,
                                                                  cell,
                                                                  num_encoder_symbols=self.n_input_symbols,
                                                                  num_decoder_symbols=self.n_output_symbols,
                                                                  embedding_size=1000,
                                                                  feed_previous=feed_previous)
        elif self.seq2seq_model == "embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                        # encoder_inputs: A list of 2D Tensors [batch_size, input_size].
                                                                        decoder_inputs,
                                                                        cell,
                                                                        num_encoder_symbols=self.n_input_symbols,
                                                                        num_decoder_symbols=self.n_output_symbols,
                                                                        embedding_size=1000,
                                                                        num_heads=4,
                                                                        initial_state_attention=False,
                                                                        feed_previous=feed_previous)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + "seq2seq_model",
                             model_outputs)  # for TFLearn to know what to save and restore
        self.net = tf.stack(model_outputs,
                            axis=1)  # shape [-1, n_decoder_inputs (= self.out_seq_len), num_decoder_symbols]

        with tf.name_scope("TargetsData"):  # placeholder for target variable (i.e. trainY input)
            targetY = tf.placeholder(shape=[None, self.out_seq_len], dtype=tf.int32, name="Y")

        self.net = tflearn.regression(self.net,
                                      placeholder=targetY,
                                      optimizer='adam',
                                      learning_rate=0.001,
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

    def getCell(self, size):
        return rnn.GRUCell(size)

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
        then use seq2seq.sequence_loss to actually compute the loss function.
        '''
        logits = tf.unstack(y_pred, axis=1)  # list of [-1, num_decoder_synbols] elements
        targets = tf.unstack(y_true,
                             axis=1)  # y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]
        sl = seq2seq.sequence_loss(logits, targets, weights)
        return sl

    def accuracy(self, y_pred, y_true,
                 x):  # y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
        '''
        Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
        values.
        '''
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))  # [-1, self.out_seq_len]
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32), name='acc')
        return accuracy

    def load(self, token_lists, model_file):
        self.prepare_data(token_lists)
        print(self.string_to_number)
        self.create_network()
        self.model.load(model_file)

    def train(self, token_lists, model_file):
        (xs, ys) = self.prepare_data(token_lists)
        self.create_network()
        # self.model.load(model_file)
        self.model.fit(xs, ys, n_epoch=1, batch_size=4096, shuffle=True, show_metric=False)
        self.model.save(model_file)
        self.model.load(model_file)

    def query(self, prefix, suffix):
        x = self.prepare_query_inputs(prefix, suffix)
        result = []
        y = self.model.predict([x])
        predicted_seq = y[0]
        for t in predicted_seq:
            if type(t) is numpy.ndarray:
                t = t.tolist()
            best_number = t.index(max(t))
            result.append(best_number)
        final = []
        if self.string_to_number["UNK"] in result:
            result = result[:result.index(self.string_to_number["UNK"])-1]
        else:
            print("Cannot find <EOS>")
            result = result[:3]

        for best_number in result:
            best_string = self.number_to_string[best_number]
            best_token = self.string_to_token(best_string)
            final.append(best_token)

        return final

    def prepare_query_inputs(self, prefix, suffix):
        x = []
        x1 = prefix[-self.window:]
        x2 = suffix[:self.window]
        for t in x1:
            x.append(self.string_to_number[self.token_to_string(t)])
        x.append(self.string_to_number["HOLE"])
        for t in x2:
            x.append(self.string_to_number[self.token_to_string(t)])
        x = [self.string_to_number["ZERO"]] * (self.window - len(x1)) + x + [self.string_to_number["ZERO"]] * (
        self.window - len(x2))
        return x

    def write_parallel_text(self, sources, targets, output_prefix):
        """
        Writes two files where each line corresponds to one example
          - [output_prefix].sources.txt
          - [output_prefix].targets.txt

        Args:
          sources: Iterator of source strings
          targets: Iterator of target strings
          output_prefix: Prefix for the output file
        """
        try:
            os.makedirs(output_prefix)
        except OSError:
            if not os.path.isdir(output_prefix):
                raise
        source_filename = os.path.abspath(os.path.join(output_prefix, "sources.txt"))
        target_filename = os.path.abspath(os.path.join(output_prefix, "targets.txt"))

        with io.open(source_filename, "w", encoding='utf8') as source_file:
            for i, record in enumerate(sources):
                src = " ".join(str(x) for x in record)
                source_file.write(src + "\n")
        print("Wrote {}".format(source_filename))

        with io.open(target_filename, "w", encoding='utf8') as target_file:
            for record in targets:
                trgt = " ".join(str(x) for x in record)
                target_file.write(trgt + "\n")
        print("Wrote {}".format(target_filename))