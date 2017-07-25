import tflearn
import tensorflow as tf
from tensorflow.contrib import legacy_seq2seq as seq2seq
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn.python.ops import core_rnn
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.ops import array_ops


class Models:
    def create_network(self, in_vocab_size, out_vocab_size, model_name="bidirectional_attention_rnn",
                       in_seq_len=15, out_seq_len=1, num_layers=4, memory_size=128, embedding_size=200, num_heads=4, scope="asdl"):

        GO_VALUE = out_vocab_size + 1

        def get_cell(size):
            return rnn.GRUCell(size)

        def cell_layers(layers=num_layers, mem=memory_size):
            cells = []
            for _ in range(layers):
                cells.append(get_cell(mem))
            return rnn.MultiRNNCell(cells)

        net = tflearn.input_data(shape=[None, in_seq_len], dtype=tf.int32, name=scope+"XY")
        encoder_inputs = tf.slice(net, [0, 0], [-1, in_seq_len], name=scope+"enc_in")

        # transform to list of self.in_seq_len elements, each [-1]
        encoder_inputs = tf.unstack(encoder_inputs, axis=1)

        decoder_inputs = tf.slice(net, [0, 0], [-1, out_seq_len], name=scope+"dec_in")
        # transform to list of self.in_seq_len elements, each [-1]
        decoder_inputs = tf.unstack(decoder_inputs, axis=1)

        # insert GO as first; drop last decoder input
        go_input = tf.multiply(tf.ones_like(decoder_inputs[0], dtype=tf.int32), GO_VALUE)
        decoder_inputs = [go_input] + decoder_inputs[: out_seq_len - 1]

        n_input_symbols = in_vocab_size + 1
        n_output_symbols = out_vocab_size + 2  # extra "GO" symbol for decoder inputs

        if model_name == "bidirectional_attention_rnn":
            model_outputs, states = self.embedding_attention_bidirectional_seq2seq(encoder_inputs,
                                                                                   decoder_inputs,
                                                                                   cell_layers(),
                                                                                   cell_layers(),
                                                                                   cell_layers(),
                                                                                   num_encoder_symbols=n_input_symbols,
                                                                                   num_decoder_symbols=n_output_symbols,
                                                                                   embedding_size=embedding_size,
                                                                                   num_heads=num_heads,
                                                                                   feed_previous=True,
                                                                                   scope=scope)
        elif model_name == "embedding_attention":
            model_outputs, states = seq2seq.embedding_attention_seq2seq(encoder_inputs,
                                                                        decoder_inputs,
                                                                        cell_layers(),
                                                                        num_encoder_symbols=n_input_symbols,
                                                                        num_decoder_symbols=n_output_symbols,
                                                                        embedding_size=embedding_size,
                                                                        num_heads=num_heads,
                                                                        initial_state_attention=False,
                                                                        feed_previous=True)
        else:
            raise Exception('[TFLearnSeq2Seq] Unknown seq2seq model %s' % self.seq2seq_model)

        # for TFLearn to know what to save and restore
        tf.add_to_collection(tf.GraphKeys.LAYER_VARIABLES + '/' + scope + "_seq2seq_model", model_outputs)

        # shape [-1, n_decoder_inputs (= self.out_seq_len), num_decoder_symbols]
        net = tf.stack(model_outputs, axis=1)

        # placeholder for target variable (i.e. trainY input)
        with tf.name_scope(scope + "TargetsData"):
            targetY = tf.placeholder(shape=[None, out_seq_len], dtype=tf.int32, name=scope+"Y")

        net = tflearn.regression(net, placeholder=targetY,
                                      optimizer='adam',
                                      learning_rate=0.01,
                                      loss=self.sequence_loss,
                                      metric=self.accuracy,
                                      name=scope+"Y")
        return tflearn.DNN(net)

    def sequence_loss(self, y_pred, y_true):
        '''
        Loss function for the seq2seq RNN.  Reshape predicted and true (label) tensors, generate dummy weights,
        then use seq2seq.sequence_loss to actually compute the loss function.
        '''
        logits = tf.unstack(y_pred, axis=1)  # list of [-1, num_decoder_synbols] elements
        targets = tf.unstack(y_true,
                             axis=1)  # y_true has shape [-1, self.out_seq_len]; unpack to list of self.out_seq_len [-1] elements

        # ones = tf.ones_like(y_true, dtype=tf.float32)
        # zeros = tf.zeros_like(y_true, dtype=tf.float32)
        # zerosint = tf.zeros_like(y_true, dtype=tf.int32)
        # weights = tf.where( tf.greater(y_true, zerosint), ones, zeros)
        # weights = tf.unstack(weights, axis=1)
        weights = [tf.ones_like(yp, dtype=tf.float32) for yp in targets]

        sl = seq2seq.sequence_loss(logits, targets, weights)

        return sl

    # y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
    def accuracy(self, y_pred, y_true, x):
        '''
        Compute accuracy of the prediction, based on the true labels.  Use the average number of equal
        values.
        '''
        # y_pred is [-1, self.out_seq_len, num_decoder_symbols]; y_true is [-1, self.out_seq_len]
        pred_idx = tf.to_int32(tf.argmax(y_pred, 2))  # [-1, self.out_seq_len]
        accuracy = tf.reduce_mean(tf.cast(tf.equal(pred_idx, y_true), tf.float32))
        return accuracy

    def embedding_attention_bidirectional_seq2seq(self, encoder_inputs, decoder_inputs, input_cell1, input_cell2,
                                                  output_cell,
                                                  num_encoder_symbols,
                                                  num_decoder_symbols, embedding_size, num_heads=4,
                                                  output_projection=None, feed_previous=False, dtype=None, scope=None,
                                                  initial_state_attention=False):

        with tf.variable_scope(scope or "embedding_attention_bidirectional_seq2seq") as scope:
            # Encoder.
            encoder_cell1 = core_rnn_cell.EmbeddingWrapper(input_cell1, embedding_classes=num_encoder_symbols,
                                                           embedding_size=embedding_size)
            encoder_cell2 = core_rnn_cell.EmbeddingWrapper(input_cell2, embedding_classes=num_encoder_symbols,
                                                           embedding_size=embedding_size)

            encoder_outputs, encoder_state1, encoder_state2 = core_rnn.static_bidirectional_rnn(encoder_cell1,
                                                                                                encoder_cell2,
                                                                                                encoder_inputs,
                                                                                                dtype=tf.float32)

            # First calculate a concatenation of encoder outputs to put attention on.
            top_states = [array_ops.reshape(e, [-1, 1, input_cell1.output_size + input_cell2.output_size]) for e in
                          encoder_outputs]

            attention_states = array_ops.concat(top_states, 1)
            encoder_state = encoder_state1 + encoder_state2

            # Decoder.
            output_size = None
            if output_projection is None:
                output_cell = rnn.OutputProjectionWrapper(output_cell, num_decoder_symbols)
                output_size = num_decoder_symbols

            assert isinstance(feed_previous, bool)
            return seq2seq.embedding_attention_decoder(decoder_inputs, encoder_state, attention_states,
                                                       output_cell,
                                                       num_decoder_symbols, embedding_size, num_heads=num_heads,
                                                       output_size=output_size,
                                                       output_projection=output_projection,
                                                       feed_previous=feed_previous,
                                                       initial_state_attention=initial_state_attention)
