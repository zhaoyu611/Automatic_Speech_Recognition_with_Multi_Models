import glob
import numpy as np
import os

def read_ndarray_from(file_dir):
    """ read numpy data from data path
        args:
            file_dir: [str] file directory
        return:
            data_list: [list]  a list of numpy data        
    """
    data_list = []
    for fname in glob.glob(os.path.join(file_dir, "*.npy")):
        data_list.append(np.load(fname))
    return data_list

def get_max_timestep(train_data, test_data):
    """ find the max timestep in train and test data
        args:
            train_data: [list] a list of train data
            test_data: [list] a list of test data
        return:
            max_timestep: [int] max time step
    """
    data_list = train_data + test_data
    max_timestep = 0
    for data in data_list:
        if data.shape[1]>max_timestep:
            max_timestep = data.shape[1]
    return max_timestep


def sparse_tuple_from(sequences, dtype=np.int32):
    """Create a sparse representention of x.
    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)
    """
    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n]*len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1]+1], dtype=np.int64)

    return indices, values, shape

import utils
if __name__ == "__main__":

    train_data_dir = "./data/TIMIT/phn/train/mfcc/"
    train_label_dir = "./data/TIMIT/phn/train/label/"
    test_data_dir = "./data/TIMIT/phn/test/mfcc/"
    test_label_dir = "./data/TIMIT/phn/test/label/"

    #each one is a list of 2D ([feature_num, time_step]) numpy data
    train_data = utils.read_ndarray_from(train_data_dir) 
    train_label = utils.read_ndarray_from(train_label_dir)
    test_data = utils.read_ndarray_from(test_data_dir)
    test_label = utils.read_ndarray_from(test_label_dir)
        
    # print(get_max_timestep(train_data, test_data))
    
    indices, values, shape = sparse_tuple_from(test_label)

