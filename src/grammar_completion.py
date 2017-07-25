from collections import defaultdict

import numpy
import tensorflow as tf
import io
import os

from models import Models


class Grammar_Completion:
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        if string == "<MARK>" or string == "<>" or string == "zend":
            return {"type": "", "value": ""}
        else:
            splitted = string.split("-@@-")
            return {"type": splitted[0], "value": splitted[1]}

    def prepare_data(self, token_lists):
        all_token_strings = set()
        token_frequencies = defaultdict(int)
        all_tokens = 0
        for token_list in token_lists:
            for token in token_list:
                all_tokens += 1
                token_frequencies[self.token_to_string(token)] += 1
                all_token_strings.add(self.token_to_string(token))

        all_token_strings.add("zend")
        all_token_strings.add("<>")
        all_token_strings.add("<MARK>")
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0

        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        self.window = 30
        self.in_seq_len = self.window * 2 + 1
        self.out_seq_len = 6
        self.in_max_int = self.out_max_int = len(self.string_to_number)

    def getTrainData(self, token_lists):
        # prepare x,y pairs
        xs = []
        ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                token_list[idx] = self.string_to_number[self.token_to_string(token)]

        z = [self.string_to_number["<>"]]
        w = self.window

        for token_list in token_lists:
            token_list = z * w + token_list + z * w
            for idx, token in enumerate(token_list):
                if idx < len(token_list) - self.in_seq_len:
                    xs.append(self.get_x_with_hole(token_list, idx, 1))
                    ys.append(self.get_y_with_hole(token_list, idx, 1))
                if idx < len(token_list) - self.in_seq_len - 1:
                    xs.append(self.get_x_with_hole(token_list, idx, 2))
                    ys.append(self.get_y_with_hole(token_list, idx, 2))
                if idx < len(token_list) - self.in_seq_len - 2:
                    xs.append(self.get_x_with_hole(token_list, idx, 3))
                    ys.append(self.get_y_with_hole(token_list, idx, 3))
                if idx < len(token_list) - self.in_seq_len - 3:
                    xs.append(self.get_x_with_hole(token_list, idx, 4))
                    ys.append(self.get_y_with_hole(token_list, idx, 4))
                if idx < len(token_list) - self.in_seq_len - 4:
                    xs.append(self.get_x_with_hole(token_list, idx, 5))
                    ys.append(self.get_y_with_hole(token_list, idx, 5))

        print("prefix x,y pairs: ", str(len(xs)), str(len(ys)))

        # xs = xs[2048000:5120000]
        # ys = ys[2048000:5120000]

        # xs = xs[4096000:]
        # ys = ys[4096000:]
        # xs = numpy.array(xs)
        # ys = numpy.array(ys)

        # print(numpy.shape(xs), numpy.shape(ys))

        self.write_parallel_text(xs, ys, "./network_inputs/dev/")

        return xs, ys

    def get_x_with_hole(self, ip, idx, hole_size):
        m = [self.string_to_number["<MARK>"]]
        w = self.window

        return ip[idx:idx + w] + m + ip[idx + w + hole_size:idx + w + w + hole_size]

    def get_y_with_hole(self, ip, idx, hole_size):
        e = [self.string_to_number["zend"]]
        z= [self.string_to_number["<>"]]
        w = self.window
        return ip[idx + w:idx + w + hole_size] + e + (z * (self.out_seq_len-hole_size -1))

    def load(self, token_lists, model_file):
        self.model_file = "./trained_model/random4/randomhole.tfl"
        self.prepare_data(token_lists)
        xs, ys = self.getTrainData(token_lists)

        # with tf.Graph().as_default():
        #     self.model = Models().create_network(self.in_max_int,
        #                                          self.out_max_int,
        #                                          model_name="bidirectional_attention_rnn",
        #                                          in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
        #                                          num_layers=2, memory_size=32,
        #                                          embedding_size=128, num_heads=4, scope="randomhole")
        #
        #     self.model.load(self.model_file)

    def train(self, token_lists, model_file):
        self.model_file = "./trained_model/random4/randomhole.tfl"
        self.prepare_data(token_lists)

        xs, ys = self.getTrainData(token_lists)
        # xs = xs[:40960]
        # ys = ys[:40960]

        with tf.Graph().as_default():
            self.model = Models().create_network(self.in_max_int, self.out_max_int,
                                                 model_name="bidirectional_attention_rnn",
                                                 in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
                                                 num_layers=2, memory_size=32,
                                                 embedding_size=128, num_heads=4, scope="randomhole")
            self.model.load(self.model_file)
            self.model.fit(xs, ys, n_epoch=1, batch_size=512, shuffle=True, show_metric=False,
                           run_id="Random Hole Completion")
            self.model.save(self.model_file)

    def query(self, prefix, suffix, expected):
        x = self.prepare_query_inputs(prefix, suffix)
        result = []
        y = self.model.predict([x])
        expected = [self.string_to_number[self.token_to_string(t)] for t in expected]
        predicted_seq = y[0]
        for t in predicted_seq:
            if type(t) is numpy.ndarray:
                t = t.tolist()
            best_number = t.index(max(t))
            result.append(best_number)
        final = []
        # if self.string_to_number["zend"] in result:
        #     result = result[:result.index(self.string_to_number["zend"])]
        # else:
        #     print("Cannot find zend")
        #     result = result[:4]
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
        x.append(self.string_to_number["<MARK>"])
        for t in x2:
            x.append(self.string_to_number[self.token_to_string(t)])
        x = [self.string_to_number["<>"]] * (self.window - len(x1)) + x + [self.string_to_number["<>"]] * (
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
            for record in sources:
                source_file.write(" ".join(str(x) for x in record) + "\n")
        print("Wrote {}".format(source_filename))
        with io.open(target_filename, "w", encoding='utf8') as target_file:
            for record in targets:
                target_file.write(" ".join(str(x) for x in record) + "\n")
