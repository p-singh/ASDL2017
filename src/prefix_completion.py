import numpy
from models import Models
import copy
from collections import defaultdict
import tensorflow as tf


class Prefix_completion:
    def token_to_string(self, token):
        return token["type"] + "-@@-" + token["value"]

    def string_to_token(self, string):
        if string == "HOLE" or string == "ZERO" or string == "UNK":
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

        all_token_strings.add("UNK")
        all_token_strings.add("ZERO")
        all_token_strings.add("HOLE")
        all_token_strings = list(all_token_strings)
        all_token_strings.sort()
        self.string_to_number = dict()
        self.number_to_string = dict()
        max_number = 0

        for token_string in all_token_strings:
            self.string_to_number[token_string] = max_number
            self.number_to_string[max_number] = token_string
            max_number += 1

        self.window = 40
        self.in_seq_len = self.window
        self.out_seq_len = 1
        self.in_max_int = self.out_max_int = len(self.string_to_number)

    def getTrainData(self, token_lists):
        # prepare x,y pairs
        prefix_xs = []
        prefix_ys = []
        suffix_xs = []
        suffix_ys = []
        for token_list in token_lists:
            for idx, token in enumerate(token_list):
                token_list[idx] = self.string_to_number[self.token_to_string(token)]

        for token_list in token_lists:
            token_list = [self.string_to_number["ZERO"]] * self.window + token_list + [self.string_to_number[
                                                                                           "ZERO"]] * self.window
            for idx, token in enumerate(token_list):
                if token_list[idx] != self.string_to_number["ZERO"]:
                    prefix_xs.append(token_list[idx-self.window:idx])
                    prefix_ys.append([token_list[idx]])
                    suffix_xs.append(token_list[idx + 1:idx + self.window + 1])
                    suffix_ys.append([token_list[idx]])

        print("prefix x,y pairs: ", str(len(prefix_xs)), str(len(prefix_ys)))
        print("suffix x,y pairs: ", str(len(suffix_xs)), str(len(suffix_ys)))

        prefix_xs = numpy.array(prefix_xs)
        prefix_ys = numpy.array(prefix_ys)
        suffix_xs = numpy.array(suffix_xs)
        suffix_ys = numpy.array(suffix_ys)
        return prefix_xs, prefix_ys, suffix_xs, suffix_ys

    def load(self, token_lists, model_file):
        self.total = 0
        self.correct = 0
        self.incorrect = 0
        self.top2 = 0
        self.top3 = 0
        self.prefix_model_file = "./trained_model/prefix_final/prefix.tfl"
        self.suffix_model_file = "./trained_model/suffix_final/suffix.tfl"
        self.prepare_data(token_lists)

        with tf.Graph().as_default():
            self.prefix_model = Models().create_network(len(self.string_to_number), len(self.string_to_number),
                                                        model_name="bidirectional_attention_rnn",
                                                        in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
                                                        num_layers=2, memory_size=64,
                                                        embedding_size=128, num_heads=8, scope="prefix")

            self.prefix_model.load(self.prefix_model_file)

        with tf.Graph().as_default():
            self.suffix_model = Models().create_network(len(self.string_to_number), len(self.string_to_number),
                                                        model_name="bidirectional_attention_rnn",
                                                        in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
                                                        num_layers=2, memory_size=64,
                                                        embedding_size=128, num_heads=8, scope="suffix")
            self.suffix_model.load(self.suffix_model_file)

    def train(self, token_lists, model_file):
        self.total = 0
        self.correct = 0
        self.incorrect = 0
        self.top2 = 0
        self.top3 = 0
        self.prefix_model_file = "./trained_model/prefix_final/prefix.tfl"
        self.suffix_model_file = "./trained_model/suffix_final/suffix.tfl"
        self.prepare_data(token_lists)
        # xs = xs[:40960]
        # ys = ys[:40960]

        (pxs, pys, sxs, sys) = self.getTrainData(token_lists)

        with tf.Graph().as_default():
            self.prefix_model = Models().create_network(len(self.string_to_number), len(self.string_to_number),
                                                        model_name="bidirectional_attention_rnn",
                                                        in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
                                                        num_layers=2, memory_size=64,
                                                        embedding_size=128, num_heads=8, scope="prefix")


            self.prefix_model.load(self.prefix_model_file)
            self.prefix_model.fit(pxs, pys, n_epoch=1, batch_size=512, shuffle=True, show_metric=False, run_id="Prefix Completion")
            self.prefix_model.save(self.prefix_model_file)

        with tf.Graph().as_default():
            self.suffix_model = Models().create_network(len(self.string_to_number), len(self.string_to_number),
                                                        model_name="bidirectional_attention_rnn",
                                                        in_seq_len=self.in_seq_len, out_seq_len=self.out_seq_len,
                                                        num_layers=2, memory_size=64,
                                                        embedding_size=128, num_heads=8, scope="suffix")
            self.suffix_model.load(self.suffix_model_file)
            self.suffix_model.fit(sxs, sys, n_epoch=1, batch_size=512, shuffle=True, show_metric=False, run_id="Suffix Completion")
            self.suffix_model.save(self.suffix_model_file)

    # Pass expeted result to this method, just ofr top 3 accuracy calculations
    def query(self, prefix, suffix, expected):
        sx = []
        x1 = suffix[:self.window]
        for t in x1:
            sx.append(self.string_to_number[self.token_to_string(t)])
        sx = sx + [self.string_to_number["ZERO"]] * (self.window - len(x1))
        turn = True

        px = []
        x2 = prefix[-self.window:]
        for t in x2:
            px.append(self.string_to_number[self.token_to_string(t)])
        px = px + [self.string_to_number["ZERO"]] * (self.window - len(x2))


        y1 = self.prefix_model.predict([px])
        y2 = self.suffix_model.predict([sx])

        self.total += 1

        predicted_seq = y1[0]
        for t1 in predicted_seq:
            if type(t1) is numpy.ndarray:
                t1 = t1.tolist()
            answers1 = copy.deepcopy(t1)
            answers1.sort(reverse=True)
            result1 = [t1.index(answers1[0]), t1.index(answers1[1]), t1.index(answers1[2])]

        predicted_seq2 = y2[0]
        for t2 in predicted_seq2:
            if type(t2) is numpy.ndarray:
                t2 = t2.tolist()
            answers2 = copy.deepcopy(t2)
            answers2.sort(reverse=True)
            result2 = [t2.index(answers2[0]), t2.index(answers2[1]), t2.index(answers2[2])]

        last_predicted = result1[0]
        predicted = result2[0]
        hole_count = 1
        self.turn = True
        while predicted != last_predicted and hole_count < 5:
            if self.turn:
                result2, c2 = self.get_prediction((result2 + sx)[:self.window])
                predicted = result2[0]
                last_predicted
            else:
                result1, c1 = self.get_prediction((px + result1)[-self.window:])


        true_val = self.string_to_number[self.token_to_string(expected[0])]
        result = [self.match_predictions(result1, result2, answers1, answers2)]
        if result[0] != true_val and (true_val == result1[0] or true_val == result2[0]):
            self.top3 +=1
        if result[0] != true_val and (true_val != result1[0] and result1[0] in result2):
            self.top2 += 1
        if result[0] != true_val and (true_val == result1[0] and true_val in result2):
            self.correct += 1

        if result1[0] == result2[0]:
            result = result1
        else:
            result = [88]

        final = []

        for best_number in result:
            best_string = self.number_to_string[best_number]
            best_token = self.string_to_token(best_string)
            final.append(best_token)
        print(result[0], "True Value: ", true_val, result1, result2, answers1[:3], answers2[:3])
        print("accuracy: ", self.correct , "top 2 acc: ", self.top2, "top3 acc: ",
              self.top3)
        return final

    def match_predictions(self, s, p, c1, c2):
        if p[0] in s:
            return p[0]
        elif s[0] in p:
            return s[0]

    def get_prediction(self, ip):
        if self.turn:
            y = self.suffix_model.predict(ip)[0][0]
            self.turn = not self.turn
        else:
            y = self.prefix_model.predict(ip)[0][0]
            self.turn = not self.turn
        if type(y) is numpy.ndarray:
            y = y.tolist()
        confidence = copy.deepcopy(y)
        confidence.sort(reverse=True)
        result = [y.index(confidence[0]), y.index(confidence[1]), y.index(confidence[2])]
        return result, confidence[:3]
