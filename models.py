import tensorflow as tf
from cells import select_cell

class Stack_Layers_Model(object):
    def __init__(self, args, inputs, targets, seq_len):
        """
            args:
                args: all argument from main.py
                inputs: [tensor] a placeholder of input, 
                                 which has shape: [batch_size, max_timesteps, num_feature]
                targets: [tensor] a palceholder of targets,
                                 which is a sparse tensor with shape: [index, value, shape]
                seq_len: [tensor] a placeholder of sequence length, which has shape: [batch_size]

        """
        self.args = args
        self.inputs = inputs
        self.targets = targets
        self.seq_len = seq_len


    def build_model(self):
        model_type = self.args.model_type
        if model_type == 'unidirection':
            return self.stack_unidirectional_rnn()
        elif model_type == 'bidirection':
            return self.stack_bidirectional_rnn()
        else:
            raise TypeError('you should specify correct model type!') 




    def stack_unidirectional_rnn(self):
        """
            stack multi-layer of unidirectional rnn
        """
        with tf.variable_scope("stack_uni-rnn"):
            stack_cells = []
            for i in range(self.args.num_layer):
                cell = select_cell(self.args)
                stack_cells.append(cell)

            mul_cells = tf.contrib.rnn.MultiRNNCell(stack_cells)
            #use dynamic rnn to get output lists and deprecated the last state
            #output shape: [batch_size, time_steps, num_hidden]
            targets, _ = tf.nn.dynamic_rnn(mul_cells, self.inputs, self.seq_len, dtype=tf.float32)
            #define full connect layer
            logits = tf.layers.dense(targets, self.args.num_class)
        return logits

    def stack_bidirectional_rnn(self):
        """
            stack multi-layer of bidirectional rnn
        """
        with tf.variable_scope("stack_bi-rnn"):
            pre_layer = self.inputs
            for i in range(self.args.num_layer):
                with tf.variable_scope("layer_{}".format(i)):
                    cell_fw = select_cell(self.args)
                    cell_bw = select_cell(self.args)
                    targets, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, pre_layer, self.seq_len, dtype=tf.float32) 
                    pre_layer = tf.concat(targets, 2) #concat the num_feature of cell_fw and cell_bw
            #define full connect layer
            logits = tf.layers.dense(pre_layer, self.args.num_class)
        return logits



