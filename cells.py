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



class HyperLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, num_units, forget_bias=1.0, use_layer_norm=True, 
                 use_recurrent_dropout=False, dropout_keep_prob=0.90,
                 hyper_num_units=32, hyper_embedding_size=16):
        """
            arg: 
                num_units: [int] hidden units num for main network
                forget_bias: [float] forget_bias
                use_layer_norm: [bool] whether use layer norm for main netwrok
                user_drop_out: [bool] whether use drop out for main network
                drop_out_pro: [float] probability of drop out
                hyper_num_units: [int] hidden units num for hyper network 
                hyper_embedding_size: [int] output units num for hyper network,
                                            always smaller than hyper_num_units
        """
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.use_layer_norm = use_layer_norm
        self.use_recurrent_dropout = use_recurrent_dropout
        self.dropout_keep_prob = dropout_keep_prob
        self.hyper_num_units = hyper_num_units
        self.hyper_embedding_size = hyper_embedding_size

        #define training units num, 
        #both hidden units in main netowrk and hyper network
        self.total_num_units = self.num_units + self.hyper_num_units

        #define hyper cell 
        self.hyper_cell = tf.contrib.rnn.BasicLSTMCell(hyper_num_units)


    @property
    def output_size(self):
        return self.num_units
    @property
    def state_size(self):
        return tf.contrib.rnn.LSTMStateTuple(self.num_units + self.hyper_num_units, 
                                             self.num_units + self.hyper_num_units)

    def hyper_norm(self, layer, hyper_output, embedding_size, num_units,
               scope="hyper", use_bias=True):
        """
            args:
                layer: [tensor] input tensor need to be normed, with shape: [batch_size, num_units]
                hyper_output: [tensor] output tensor with shape: [batch_size, embedding_size] 
                num_units: [int] num of main network's hidden units

                init_gamma= 0.10
        """
        init_gamma = 0.10

        with tf.variable_scope(scope):
            with tf.variable_scope('zw'):
                zw = _linear(hyper_output, embedding_size, False)
            with tf.variable_scope('alpha'):    
                alpha = _linear(zw, num_units, False)

            result = tf.multiply(alpha, layer)

        return result


    def hyper_bias(self, layer, hyper_output, embedding_size, num_units, 
                   scope="hyper"):
        with tf.variable_scope(scope):
            with tf.variable_scope('zb'):
                zb = _linear(hyper_output, embedding_size, False)
            with tf.variable_scope('beta'):
                beta = _linear(zb, num_units, False)

        return layer + beta


    def __call__(self, x, state, scope=None):
        """
            arg: 
                x: [tensor] input tensor at each time step with shape: [batch_szie, num_input]
                state: [tuple] state at last time step 
        """
        with tf.variable_scope(scope or type(self).__name__):
            total_c, total_h = state
            c = total_c[:, 0:self.num_units]
            h = total_h[:, 0:self.num_units]
            hyper_state = tf.contrib.rnn.LSTMStateTuple(total_c[:, self.num_units:],
                                                        total_h[:, self.num_units:])
            x_size = x.get_shape().as_list()[1]
            batch_size = x.get_shape().as_list()[0]
            embedding_size = self.hyper_embedding_size
            num_units = self.num_units

            #define weights and bias for main network
            # W_xh = tf.get_variable('W_xh', initializer=tf.random_normal([x_size, 4*num_units]), dtype=tf.float32)
            # W_hh = tf.get_variable('W_hh', initializer=tf.random_normal([num_units, 4*num_units]), dtype=tf.float32)
            # bias = tf.get_variable('bias', initializer=tf.random_normal([4*num_units]), dtype=tf.float32)
            W_xh_val = np.random.normal(size=[x_size, 4*num_units])
            w_hh_val = np.random.normal(size=[num_units, 4*num_units])
            W_xh = tf.get_variable('W_xh', shape=[x_size, 4*num_units], initializer=tf.constant_initializer(W_xh_val))
            W_hh = tf.get_variable('W_hh', shape=[num_units, 4*num_units],initializer=tf.constant_initializer(w_hh_val))
            bias = tf.get_variable('bias', shape=[4*num_units], initializer=tf.constant_initializer(0))

            #define hyper network input, shape : [batch_size, x_size+num_units]
            hyper_input = tf.concat([x,h], 1)
            hyper_output, hyper_new_state = self.hyper_cell(hyper_input, hyper_state)

            xh = tf.matmul(x, W_xh)
            hh = tf.matmul(h, W_hh)

            #split Wxh contributions
            ix, jx, fx, ox = tf.split(xh, 4, 1)

            ix = self.hyper_norm(ix, hyper_output, embedding_size, num_units, 'hyper_ix')
            jx = self.hyper_norm(jx, hyper_output, embedding_size, num_units, 'hyper_jx')
            fx = self.hyper_norm(fx, hyper_output, embedding_size, num_units, 'hyper_fx')
            ox = self.hyper_norm(ox, hyper_output, embedding_size, num_units, 'hyper_ox')

            #split Whh contributions
            ih, jh, fh, oh = tf.split(hh, 4, 1)
            ih = self.hyper_norm(ih, hyper_output, embedding_size, num_units, 'hyper_ih')
            jh = self.hyper_norm(jh, hyper_output, embedding_size, num_units, 'hyper_jh')
            fh = self.hyper_norm(fh, hyper_output, embedding_size, num_units, 'hyper_fh')
            oh = self.hyper_norm(oh, hyper_output, embedding_size, num_units, 'hyper_oh')

            #split bias      
            ib, jb, fb, ob = tf.split(bias, 4, 0)
            ib = self.hyper_bias(ib, hyper_output, embedding_size, num_units, 'hyper_ib')
            jb = self.hyper_bias(jb, hyper_output, embedding_size, num_units, 'hyper_jb')
            fb = self.hyper_bias(fb, hyper_output, embedding_size, num_units, 'hyper_fb')
            ob = self.hyper_bias(ob, hyper_output, embedding_size, num_units, 'hyper_ob')

            #i = input_gate, j = new_input, f= forget_gate, o = output_gate
            i = ix + ih + ib
            j = jx + jh + jb
            f = fx + fh + fb
            o = ox + oh + ob

            if self.use_layer_norm:
                i = layer_norm(i, scope='ln_i/')
                j = layer_norm(j, scope='ln_j/')
                f = layer_norm(f, scope='ln_f/')
                o = layer_norm(o, scope='ln_o/')

            if self.use_recurrent_dropout:
                g = tf.nn.dropout(tf.tanh(j), self.dropout_keep_prob)
            else:
                g = tf.tanh(j)

            new_c = c*tf.sigmoid(f+self.forget_bias) + tf.sigmoid(i)*g
            if self.use_layer_norm:
                new_h = tf.tanh(layer_norm(new_c, scope='ln_c/')) * tf.sigmoid(o)
            else:
                new_h = tf.tanh(new_c) * tf.sigmoid(o)

            hyper_c, hyper_h = hyper_new_state
            new_total_c = tf.concat([new_c, hyper_c], 1)
            new_total_h = tf.concat([new_h, hyper_h], 1)

        return new_h, tf.contrib.rnn.LSTMStateTuple(new_total_c, new_total_h)         

 



def select_cell(args):
    cell_type = args.cell_type
    num_hidden = args.num_hidden
    layer_norm = args.layer_norm

    if cell_type == "LSTMCell":
        cell = LayerNormLSTMCell(num_hidden, layer_norm)
    elif cell_type == "RNNCell":
        cell = tf.contrib.rnn.BasicRNNCell(num_hidden)
    elif cell_type == "GRUCell":
        cell = tf.contrib.rnn.GRUCell(num_hidden)
    elif cell_type == "HyperLSTMCell":
        cell = HyperLSTMCell(num_hidden, use_layer_norm=layer_norm)
    else:
        raise Exception("There is no cell type: {}, check again !".format(cell_type))
    return cell 




# test if the layer normarlization works well
if __name__ == '__main__':
    #shape is [batch_size, time_steps, num_inputs]
    num_input = 28
    batch_size = 128
    time_steps = 28
    num_hidden = 128
    
    inputs = tf.placeholder(tf.float32, [batch_size, time_steps,num_input]) 
    inp = tf.placeholder(tf.float32, [batch_size, num_hidden])

    # cell = HyperLSTMCell(num_hidden)
    # print cell
    print(tf.constant_initializer(1))