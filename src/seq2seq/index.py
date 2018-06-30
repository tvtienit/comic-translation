# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
%matplotlib inline
import math
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pylab
from collections import Counter
import csv

# Seq2Seq Items
import tensorflow.contrib.seq2seq as seq2seq
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell
from tensorflow.contrib.seq2seq.python.ops import attention_wrapper
from tensorflow.python.layers.core import Dense

#------------------------------------------------------------------------------------------------------------------------------#


vocab_size= 50000
num_units = 128
input_size = 128
batch_size = 128
source_sequence_length=40
target_sequence_length=60
decoder_type = 'basic' # could be basic or attention
sentences_to_read = 50000
#------------------------------------------------------------------------------------------------------------------------------#

src_dictionary = dict()
with open('train_eng_vocabulary.txt', encoding='utf-8') as f:
    for line in f:
        #we are discarding last char as it is new line char
        src_dictionary[line[:-1]] = len(src_dictionary)

src_reverse_dictionary = dict(zip(src_dictionary.values(),src_dictionary.keys()))

print('Source')
print('\t',list(src_dictionary.items())[:10])
print('\t',list(src_reverse_dictionary.items())[:10])
print('\t','Vocabulary size: ', len(src_dictionary))

tgt_dictionary = dict()
with open('train_vi_vocabulary.txt', encoding='utf-8') as f:
    for line in f:
        #we are discarding last char as it is new line char
        tgt_dictionary[line[:-1]] = len(tgt_dictionary)

tgt_reverse_dictionary = dict(zip(tgt_dictionary.values(),tgt_dictionary.keys()))

print('Target')
print('\t',list(tgt_dictionary.items())[:10])
print('\t',list(tgt_reverse_dictionary.items())[:10])
print('\t','Vocabulary size: ', len(tgt_dictionary))

source_sent = []
target_sent = []

test_source_sent = []
test_target_sent = []


with open('train_eng.txt', encoding='utf-8') as f:
    for l_i, line in enumerate(f):
        source_sent.append(line)
        if len(source_sent)>=sentences_to_read:
            break
        
            
with open('train_vi.txt', encoding='utf-8') as f:
    for l_i, line in enumerate(f):
        target_sent.append(line)
        if len(target_sent)>=sentences_to_read:
            break
        
            
assert len(source_sent)==len(target_sent),'Source: %d, Target: %d'%(len(source_sent),len(target_sent))

print('Sample translations (%d)'%len(source_sent))
for i in range(0,sentences_to_read,10000):
    print('(',i,') EN: ', source_sent[i])
    print('(',i,') vi: ', target_sent[i])

#------------------------------------------------------------------------------------------------------------------------------#
def split_to_tokens(sent,is_source):
    #sent = sent.replace('-',' ')
    #sent = sent.replace(',',' ,')
    #sent = sent.replace('.',' .')
    sent = sent.replace('\n',' ') 
    
    sent_toks = sent.split(' ')
    for t_i, tok in enumerate(sent_toks):
        if is_source:
            if tok not in src_dictionary.keys():
                sent_toks[t_i] = '<unk>'
        else:
            if tok not in tgt_dictionary.keys():
                sent_toks[t_i] = '<unk>'
    return sent_toks

# Let us first look at some statistics of the sentences
source_len = []
source_mean, source_std = 0,0
for sent in source_sent:
    source_len.append(len(split_to_tokens(sent,True)))

print('(Source) Sentence mean length: ', np.mean(source_len))
print('(Source) Sentence stddev length: ', np.std(source_len))

target_len = []
target_mean, target_std = 0,0
for sent in target_sent:
    target_len.append(len(split_to_tokens(sent,False)))

print('(Target) Sentence mean length: ', np.mean(target_len))
print('(Target) Sentence stddev length: ', np.std(target_len))

#------------------------------------------------------------------------------------------------------------------------------#

train_inputs = []
train_outputs = []
train_inp_lengths = []
train_out_lengths = []

max_tgt_sent_lengths = 0

src_max_sent_length = 39
tgt_max_sent_length = 48
for s_i, (src_sent, tgt_sent) in enumerate(zip(source_sent,target_sent)):
    
    src_sent_tokens = split_to_tokens(src_sent,True)
    tgt_sent_tokens = split_to_tokens(tgt_sent,False)
        
    num_src_sent = []
    for tok in src_sent_tokens:
        num_src_sent.append(src_dictionary[tok])

    num_src_set = num_src_sent[::-1] # we reverse the source sentence. This improves performance
    num_src_sent.insert(0,src_dictionary['<s>'])
    train_inp_lengths.append(min(len(num_src_sent)+1,src_max_sent_length))
    
    # append until the sentence reaches max length
    if len(num_src_sent)<src_max_sent_length:
        num_src_sent.extend([src_dictionary['</s>'] for _ in range(src_max_sent_length - len(num_src_sent))])
    # if more than max length, truncate the sentence
    elif len(num_src_sent)>src_max_sent_length:
        num_src_sent = num_src_sent[:src_max_sent_length]
    assert len(num_src_sent)==src_max_sent_length,len(num_src_sent)

    train_inputs.append(num_src_sent)

    num_tgt_sent = [tgt_dictionary['</s>']]
    for tok in tgt_sent_tokens:
        num_tgt_sent.append(tgt_dictionary[tok])
    
    train_out_lengths.append(min(len(num_tgt_sent)+1,tgt_max_sent_length))
    
    if len(num_tgt_sent)<tgt_max_sent_length:
        num_tgt_sent.extend([tgt_dictionary['</s>'] for _ in range(tgt_max_sent_length - len(num_tgt_sent))])
    elif len(num_tgt_sent)>tgt_max_sent_length:
        num_tgt_sent = num_tgt_sent[:tgt_max_sent_length]
    
    train_outputs.append(num_tgt_sent)
    assert len(train_outputs[s_i])==tgt_max_sent_length, 'Sent length needs to be 60, but is %d'%len(binned_outputs[s_i])    

assert len(train_inputs)  == len(source_sent),\
        'Size of total bin elements: %d, Total sentences: %d'\
                %(len(train_inputs),len(source_sent))

print('Max sent lengths: ', max_tgt_sent_lengths)


train_inputs = np.array(train_inputs, dtype=np.int32)
train_outputs = np.array(train_outputs, dtype=np.int32)
train_inp_lengths = np.array(train_inp_lengths, dtype=np.int32)
train_out_lengths = np.array(train_out_lengths, dtype=np.int32)
print('Samples from bin')
print('\t',[src_reverse_dictionary[w]  for w in train_inputs[0,:].tolist()])
print('\t',[tgt_reverse_dictionary[w]  for w in train_outputs[0,:].tolist()])
print('\t',[src_reverse_dictionary[w]  for w in train_inputs[10,:].tolist()])
print('\t',[tgt_reverse_dictionary[w]  for w in train_outputs[10,:].tolist()])
print()
print('\tSentences ',train_inputs.shape[0])
