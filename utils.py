import glob
import numpy as np
import os
import tensorflow as tf


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



def data_lists_to_batches(inputList, targetList, batchSize, level, max_timestep):
    ''' padding the input list to a same dimension, integrate all data into batchInputs
        args:
            inputList: [list] a list of input data
            targetList: [list] a list of target data
            batchSize: [int] batch siee
            level: [str] choose 'phn' or 'cha' 
            max_timestep: [int] the max time step over both train and test dataset
        return:
            batched_data: [tuple] contains (dataBatches, maxLength)
                                  where dataBatches is a list of length (n_samples/batch_size)
                                        maxLength is an int number of the longest of each batch
    '''
    assert len(inputList) == len(targetList)
    # dimensions of inputList:batch*39*time_length

    nFeatures = inputList[0].shape[0]
    maxLength = max_timestep
    # maxLength = 0
    # for inp in inputList:
    #     # find the max time_length
    #     maxLength = max(maxLength, inp.shape[1])

    # randIxs is the shuffled index from range(0,len(inputList))
    randIxs = np.random.permutation(len(inputList))
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        # batchSeqLengths store the time-length of each sample in a mini-batch
        batchSeqLengths = np.zeros(batchSize)

        # randIxs is the shuffled index of input list
        for batchI, origI in enumerate(randIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]

        batchInputs = np.zeros((maxLength, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(randIxs[start:end]):
            # padSecs is the length of padding
            padSecs = maxLength - inputList[origI].shape[1]
            # numpy.pad pad the inputList[origI] with zeos at the tail
            batchInputs[:,batchI,:] = np.pad(inputList[origI].T, ((0,padSecs),(0,0)), 'constant', constant_values=0)
            # target label
            batchTargetList.append(targetList[origI])
        batchInputs = np.transpose(batchInputs, [1, 0, 2])
        dataBatches.append((batchInputs, list_to_sparse_tensor(batchTargetList, level), batchSeqLengths))
        start += batchSize
        end += batchSize
    return (dataBatches, maxLength)


def list_to_sparse_tensor(targetList, level):
    ''' turn 2-D List to SparseTensor
    '''
    indices = [] #index
    vals = [] #value
    assert level == 'phn' or level == 'cha', 'type must be phoneme or character, seq2seq will be supported in future'
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',\
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',\
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',\
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',\
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',\
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',\
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',\
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',\
       'v', 'w', 'y', 'z', 'zh']

    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']


    mapping = {'ah': 'ax', 'ax-h': 'ax', 'ux': 'uw', 'aa': 'ao', 'ih': 'ix', \
               'axr': 'er', 'el': 'l', 'em': 'm', 'en': 'n', 'nx': 'n',\
               'eng': 'ng', 'sh': 'zh', 'hv': 'hh', 'bcl': 'h#', 'pcl': 'h#',\
               'dcl': 'h#', 'tcl': 'h#', 'gcl': 'h#', 'kcl': 'h#',\
               'q': 'h#', 'epi': 'h#', 'pau': 'h#'}

    group_phn = ['ae', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'dx', 'eh', \
                 'er', 'ey', 'f', 'g', 'h#', 'hh', 'ix', 'iy', 'jh', 'k', 'l', \
                 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 't', 'th', 'uh', 'uw',\
                 'v', 'w', 'y', 'z', 'zh']

    if level == 'cha':
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(axis=0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    elif level == 'phn':
        '''
        for phn level, we should collapse 61 labels into 39 labels before scoring
        
        Reference:
          Heterogeneous Acoustic Measurements and Multiple Classifiers for Speech Recognition(1986), 
            Andrew K. Halberstadt, https://groups.csail.mit.edu/sls/publications/1998/phdthesis-drew.pdf
        '''
        for tI, target in enumerate(targetList):
            for seqI, val in enumerate(target):
                if val < len(phn) and (phn[val] in mapping.keys()):
                    val = group_phn.index(mapping[phn[val]])
                indices.append([tI, seqI])
                vals.append(val)
        shape = [len(targetList), np.asarray(indices).max(0)[1]+1] #shape
        return (np.array(indices), np.array(vals), np.array(shape))

    else:
        ##support seq2seq in future here
        raise ValueError('Invalid level: %s'%str(level))


def get_edit_distance(hyp_arr, truth_arr, normalize, level):
    ''' calculate edit distance
    This is very universal, both for cha-level and phn-level
    '''

    graph = tf.Graph()
    with graph.as_default():
        truth = tf.sparse_placeholder(tf.int32)
        hyp = tf.sparse_placeholder(tf.int32)
        editDist = tf.reduce_sum(tf.edit_distance(hyp, truth, normalize=normalize))

    with tf.Session(graph=graph) as session:
        truthTest = list_to_sparse_tensor(truth_arr, level)
        hypTest = list_to_sparse_tensor(hyp_arr, level)
        feedDict = {truth: truthTest, hyp: hypTest}
        dist = session.run(editDist, feed_dict=feedDict)
    return dist



def output_to_sequence(lmt, type='phn'):
    ''' convert the output into sequences of characters or phonemes
    '''
    phn = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h',
       'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng',
       'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k',
       'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow',
       'oy', 'p', 'pau', 'pcl', 'q', 'r', 's',
       'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux',
       'v', 'w', 'y', 'z', 'zh']
    sequences = []
    start = 0
    sequences.append([])
    for i in range(len(lmt[0])):
        if lmt[0][i][0] == start:
            sequences[start].append(lmt[1][i])
        else:
            start = start + 1
            sequences.append([])

    #here, we only print the first sequence of batch
    indexes = sequences[0] #here, we only print the first sequence of batch
    if type == 'phn':
        seq = []
        for ind in indexes:
            if ind == len(phn):
                pass
            else:
                seq.append(phn[ind])
        seq = ' '.join(seq)
        return seq

    elif type == 'cha':
        seq = []
        for ind in indexes:
            if ind == 0:
                seq.append(' ')
            elif ind == 27:
                seq.append("'")
            elif ind == 28:
                pass
            else:
                seq.append(chr(ind+96))
        seq = ''.join(seq)
        return seq
    else:
        raise TypeError('mode should be phoneme or character')






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

