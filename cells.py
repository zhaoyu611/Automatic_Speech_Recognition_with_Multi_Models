import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _linear
import numpy as np

def layer_norm(inp, eps=1e-5, scope=None):
    """
        args:
            inp: [tensor] a 2D tensor with shape: [batch_size, num_hidden]
            eps: [float] for math stability
            scope: [str] variable scope
        return:
            ln_inp: [tensor] layer normed input, the same shape with 'inp'
    """
    assert (len(inp.get_shape()) == 2)
    mean, var = tf.nn.moments(inp, [1], keep_dims=True)
    scope = '' if scope == None else scope
    with tf.variable_scope(scope + 'layer_norm'):
        gain = tf.get_variable('gain', shape=[inp.get_shape()[1]], initializer=tf.constant_initializer(1))
        bias = tf.get_variable('bias', shape=[inp.get_shape()[1]], initializer=tf.constant_initializer(0))

    ln_inp = (inp - mean) / tf.sqrt(var + eps)
    ln_inp = gain * ln_inp + bias   
    return ln_inp

class LayerNormLSTMCell(tf.contrib.rnn.RNNCell):

    def __init__(self, num_units, layer_norm=True, forget_bias=1.0, activation=tf.nn.tanh):
        # forget bias is pretty important for training
        self._num_units = num_units
        self._forget_bias = forget_bias
        self._activation = activation

    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self._num_units, self._num_units)    

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state
            concat = _linear([inputs, h], 4*self._num_units, False)
            # concat = tf.layers.dense([inputs, h], 4*self._num_units)
            i,j,f,o = tf.split(concat, 4, 1)

            if layer_norm:    
                #add layer normalization for each gate before activation
                i = layer_norm(i, scope='i/')
                j = layer_norm(j, scope='j/')
                f = layer_norm(f, scope='f/')
                o = layer_norm(o, scope='o/')

            new_c = c * tf.nn.sigmoid(f + self._forget_bias) + tf.nn.sigmoid(i) * self._activation(j)
            if layer_norm:
                new_c =  layer_norm(new_c, scope='h/')
            #add layer normalization in calculation of new hidden state
            new_h = self._activation(new_c) * tf.nn.sigmoid(o)
            
            new_state = tf.nn.rnn_cell.LSTMStateTuple(new_c, new_h)

            return new_h, new_state

def select_cell(args):
    cell_type = args.cell_type
    num_hidden = args.num_hidden
    layer_norm = args.layer_norm

    if cell_type == "LSTMCell":
        cell = LayerNormLSTMCell(num_hidden, layer_norm)
    else:
        raise Exception("There is no specified cell type, check again !")
    return cell 



# def select_cell_fn(cell_type):
#     if cell_type == "LSTMCell":
#         cell_fn = LayerNormLSTMCell
#     elif cell_type == "RNNCell":
#         cell_fn = tf.contrib.rnn.BasicRNNCell #need to be redifned with layernorm or something else
#     else:
#         raise Exception("There is no specified cell type, check again !")
#     return cell_fn


# def select_cell(args):
#     if args.cell_type == "LSTMCell":
#         cell = lstm_cell(args)
#     elif args.cell_type == "GRUCell":
#         cell = GRU_cell(args)
#     elif args.cell_type == "RNN_cell":
#         cell = RNN_cell(args)
#     else:
#         raise Exception("There is no specified cell type, check again !")

#     return cell

# def lstm_cell(args):
#     num_hidden = args.num_hidden
#     return tf.contrib.rnn.LSTMCell(num_hidden)

# def GRU_cell(args):
#     num_hidden = args.num_hidden
#     return tf.contrib.rnn.GRUCell(num_hidden)

# def RNN_cell(args):
#     num_hidden = args.num_hidden
#     return tf.contrib.rnn.RNNCell(num_hidden)


