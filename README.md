# Automatic_Speech_Recognition_with_Multi_models


An Automatic Speech Recognition (ASR) Model in Tensorflow, which is inspired by repo: [Automatic-Speech-Recognition](https://github.com/zzw922cn/Automatic_Speech_Recognition). I highly recommend to read its README.md before starting this one. 

Compared to the upper repo, there are server mainly diferrences :

1. Only do experiment on a classical benchmark: [TIMIT corpus](https://catalog.ldc.upenn.edu/ldc93s1). 
2. Only use PER as cost function.
3. Focus on neural network cells and models

In sum, this repo can be regarded as tiny version from [Automatic-Speech-Recognition](https://github.com/zzw922cn/Automatic_Speech_Recognition). Therefore, it is familiar to newbie to step in ASR and deep leraning.


## Getting Started


### Prerequisites

Written in Python 2.7, and Python 3 is not tested.

This project depends on scikit.audiolab, for which you need to have [libsndfile](http://www.mega-nerd.com/libsndfile/) installed in your system.
Clone the repository to your preferred directory and install the dependencies using:
<pre>
pip install -r requirements.txt
</pre>

### TIMIT Corpus Dataset

#### Download 


Experiment is done on [TIMIT corpus](https://catalog.ldc.upenn.edu/ldc93s1). 
An avaible download url is provided by [timit](https://github.com/philipperemy/timit).

Once done, put folder 'TIMIT' at the root path of your project. Please confirm 
the folder in the same path with 'timit_preprocess.py'

#### Preprocess 

The original TIMIT database contains 6300 utterances, but we find the 'SA' audio files occurs many times, it will lead bad bias for our speech recognition system. Therefore, we removed the all 'SA' files from the original dataset and attain the new TIMIT dataset, which contains only 5040 utterances including 3696 standard training set and 1344 test set.

Automatic Speech Recognition transcribes a raw audio file into character sequences; the preprocessing stage converts a raw audio file into feature vectors of several frames. We first split each audio file into 20ms Hamming windows with an overlap of 10ms, and then calculate the 12 mel frequency ceptral coefficients, appending an energy variable to each frame. This results in a vector of length 13. We then calculate the delta coefficients and delta-delta coefficients, attaining a total of 39 coefficients for each frame. In other words, each audio file is split into frames using the Hamming windows function, and each frame is extracted to a feature vector of length 39 (to attain a feature vector of different length, modify the settings in the file [timit_preprocess.py](https://github.com/zhaoyu611/Automatic_Speech_Recognition_with_Multi_Models/blob/master/preprocess/timit/timit_preprocess.py).

In folder data/mfcc, each file is a feature matrix with size timeLength\*39 of one audio file; in folder data/label, each file is a label vector according to the mfcc file.

The original TIMIT dataset contains 61 phonemes, we use 61 phonemes for training and evaluation, but when scoring, we mappd the 61 phonemes into 39 phonemes for better performance. We do this mapping according to the paper [Speaker-independent phone recognition using hidden Markov models](http://repository.cmu.edu/cgi/viewcontent.cgi?article=2768&context=compsci). The mapping details are as follows:

| Original Phoneme(s) | Mapped Phoneme |
| :------------------  | :-------------------: |
| iy | iy |
| ix, ih | ix |
| eh | eh |
| ae | ae |
| ax, ah, ax-h | ax | 
| uw, ux | uw |
| uh | uh |
| ao, aa | ao |
| ey | ey |
| ay | ay |
| oy | oy |
| aw | aw |
| ow | ow |
| er, axr | er |
| l, el | l |
| r | r |
| w | w |
| y | y |
| m, em | m |
| n, en, nx | n |
| ng, eng | ng |
| v | v |
| f | f |
| dh | dh |
| th | th |
| z | z |
| s | s |
| zh, sh | zh |
| jh | jh |
| ch | ch |
| b | b |
| p | p |
| d | d |
| dx | dx |
| t | t |
| g | g |
| k | k |
| hh, hv | hh |
| bcl, pcl, dcl, tcl, gcl, kcl, q, epi, pau, h# | h# |

in order to preprocess TIMIT data, run the following command to see available arguments:
<pre>
python timit_preprocess.py -h

usage: timit_preprocess [-h] [-n {train,test}] [-l {cha,phn}]
                        [-m {mfcc,fbank}] [--featlen FEATLEN] [--seq2seq]
                        [-winlen WINLEN] [-winstep WINSTEP]
                        path save

Script to preprocess timit data

positional arguments:
  path                  Directory where Timit dataset is contained
  save                  Directory where preprocessed arrays are to be saved

optional arguments:
  -h, --help            show this help message and exit
  -n {train,test}, --name {train,test}
                        Name of the dataset
  -l {cha,phn}, --level {cha,phn}
                        Level
  -m {mfcc,fbank}, --mode {mfcc,fbank}
                        Mode
  --featlen FEATLEN     Features length
  --seq2seq             set this flag to use seq2seq
  -winlen WINLEN, --winlen WINLEN
                        specify the window length of feature
  -winstep WINSTEP, --winstep WINSTEP
                        specify the window step length of feature

</pre>

***It should be pointed that the preprocess code is copied from [Automatic-Speech-Recognition](https://github.com/zzw922cn/Automatic_Speech_Recognition)***

### Neural Network Cells

In theory, all cells are feasible. But the ASR is a typical 
temperal sequence classification problem. We pay more attention about LSTM and its variants. 

All available cells are stored in cells.py:

+ Basic RNN
+ Basic GRU
+ [Layer Norm LSTM](https://arxiv.org/abs/1607.06450v1) 
+ [Hyper LSTM](https://arxiv.org/abs/1609.09106) (option: with or without layer norm)

### Nerual Network Models

For ASR, most papers use CTC or seq2seq to for acoustic model. 
Currently, only partly of neural network with CTC is finished.
seq2seq will come soon. 

1. models with CTC

+ unidirectional RNN 
+ bidirectional RNN
+ Residual Network (come soon)
+ Ensemble Netowrk (come soon)

2. seq2seq (come soon)

## Special Thanks

+ zzw922cn for providing the preprocess scripts [Automatic_Speech_Recognition](https://github.com/zzw922cn/Automatic_Speech_Recognition)
+ hardmaru for providing [supercell](https://github.com/hardmaru/supercell)





