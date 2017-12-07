import tensorflow as tf



def select_cell(args):
    if args.cell_type == 'BasicLSTM':
        cell = basic_lstm(args)
    return cell

def basic_lstm(args):
    num_hidden = args.num_hidden
    return tf.contrib.rnn.BasicLSTMCell(num_hidden)






