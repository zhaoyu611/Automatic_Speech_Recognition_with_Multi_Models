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


    def stack_RNN(self):
        """ 
            stack multi-layer unidirectional or bidirectional RNN. And residual connection
            is available option.
            
            return:
                logits: [tensor] the output layer tensor with shape: [batch_size, time_step, num_feature]
        """
        isBiRNN = self.args.isBiRNN # a bool to judge bidirectional or unidirectional RNN
        scope_name = "stack_bi_RNN" if isBiRNN else "stack_uni_RNN"
        with tf.variable_scope(scope_name):
            inputs = self.inputs 
            num_layer = self.args.num_layer
            # dropout only work for the fisrt and last layers whiling training
            inputs = tf.contrib.layers.dropout(
                                                inputs, 
                                                keep_prob=(1-self.args.dropout), 
                                                is_training=self.args.isTrain)
            for i in range(num_layer):
                with tf.variable_scope("layer_{}".format(i+1)):

                    if isBiRNN: # if use bidirectional RNN
                        cell_fw = select_cell(self.args)
                        cell_bw = select_cell(self.args)
                        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, self.seq_len, dtype=tf.float32)
                        # add value of cell_fw and cell_bw in dim [num_feature]
                        outputs = tf.concat(outputs, 2) #concat the num_feature of cell_fw and cell_bw
                        shape = outputs.get_shape().as_list() #get shape: [batch_size, time_step, num_feature+num_feature]
                        outputs = tf.reshape(outputs, [shape[0], shape[1], 2, int(shape[2]/2)])
                        outputs = tf.reduce_sum(outputs, 2) #get shape: [batch_size, time_step, num_feature]
                    else: #if use unidirectional RNN
                        cell = select_cell(self.args)
                        outputs, _ = tf.nn.dynamic_rnn(cell, inputs, self.seq_len, dtype=tf.float32)
                    
                    # calculate next layer's inputs
                    if self.args.isResNet and i != 0:
                        with tf.variable_scope("ResNet_in_layer_{}".format(i+1)): 
                            assert inputs.get_shape().as_list() == outputs.get_shape().as_list(), \
                                   "Please confirm the inputs and outpus have the same dim for ResNet"
                            if self.args.num_layer < 3:
                                print("we highly recommend to use ResNet when hidden layer is larger than 2")
                            inputs = outputs + inputs  #no residual between the input and output layers
                    else:
                        inputs = outputs
            # dropout only work for the fisrt and last layers whiling training
            inputs = tf.contrib.layers.dropout(
                                                inputs, 
                                                keep_prob=(1-self.args.dropout), 
                                                is_training=self.args.isTrain) 
            logits = tf.layers.dense(inputs, self.args.num_class)
        return logits



