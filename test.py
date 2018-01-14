import tensorflow as tf
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



print(tf.__version__)
inp = tf.zeros(shape=[32, 50])
layer_norm(inp)