# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Eval pre-trained 1 billion word language model.
"""
import os
import sys

import numpy as np
from six.moves import xrange
import tensorflow as tf

from google.protobuf import text_format
import data_utils

sys.path.append("pytorch_lm/")

import cPickle as pickle
import torch
from torch.autograd import Variable

'''
1. # KENLM to train

bin/lmplz -o 5 <lm_train_data.txt >lm_model.arpa

2. #to convert arpa to binary

bin/build_binary lm_model.arpa lm_model.binary
'''

def load_RNNLM_model(checkpoint="pytorch_lm/model_small_vocab.pt", vocab_path="pytorch_lm/pytorch_lm_vocab.pkl", cuda=False):
    '''
    loads the pytorch lm model and vocab. returns the model, a vocab dict, and a word list
    '''
    with open(checkpoint, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)
        model.eval()  

    if cuda:
        model.cuda()
    else:
        model.cpu()

    # this should be a dict
    word2idx = dict(pickle.load( open(vocab_path, "rb" )))
    # this should be a list
    idx2word = [key for key, value in word2idx.iteritems()]
               
    return model, word2idx, idx2word



def get_RNNLM_softmax(model, prefix_words, word2idx, idx2word, cuda=False):
    assert prefix_words is not None and word2idx is not None and idx2word is not None, "arg to get_RNNLM_softmax cannot be none"
    
    print("prefix_words: ", prefix_words)
    
    # initialize hidden for every sample
    hidden = model.init_hidden(1)
    
    prefix_words = prefix_words.lower().strip()   
    prefix = [word2idx[x] if x in word2idx else word2idx["<unk>"] for x in prefix_words.split()]  #Convert a string to a list of tokens
    if len(prefix) <= 1:
        return torch.FloatTensor(len(idx2word)).zero_()    
    
              
    intoken = prefix[0]
    prefix = prefix[1:]
    input = Variable(torch.LongTensor([intoken]), volatile=True)  #The input to the model will be a [1,1] LongTensor with the token information, make sure volatile is True!!!!!
    
    if cuda:
        input.data = input.data.cuda()
    
    generate = False
    while True:
        output, hidden = model(input, hidden)  ##This is to run one step of the model 
    
        if not generate:
            intoken = prefix[0]  #Take the next word from a prefix
            prefix = prefix[1:] 
            input.data.fill_(intoken) #fill the input variable with the next token of the prefix and ignore the output
            if prefix == []:  #Once prefix is empty, we have fed everything to the model, the next output will be the softmax we want
                generate = True
    
        else:
            word_weights = output.squeeze().data.cpu() 
            return word_weights



def get_KenLM_Softmax(prefix_words, vocab_words, model):
    assert prefix_words is not None and vocab_words is not None, "arg to get_KenLM_Softmax cannot be none"
#    prefix_words = "this will be the tokens generated till"

    #gives prob of tokens given start                                                                 
    
    #this will be all the vocab words
#    vocab = ["now", "a", "good", "heart", "is", "all", "you", "need", "."]                            
    prefix_score = model.score(prefix_words, bos=True, eos=False)
    softmax = []                                                                                        
    for word in vocab_words:
        #assumption: does not change the in_state as we want probs for the entire vocab for the same in_state|| code says updates out_state though

        new_sent = prefix_words + " " + word
        new_score = model.score(new_sent, bos=True, eos=False)
        prob = new_score - prefix_score
        
        
        if prob < 0 and prob > 1:
            print("something wrong in KENLM")
            exit()                            

        softmax.append(10**prob)
#    print("argmax: %s and word: %s", np.argmax(softmax), vocab_words[np.argmax(softmax)]) 
    softmax = softmax / np.sum(softmax)
#    print("kenlm prob sum: ", sum(softmax))   
    return softmax



############ chucked this model as TF sucks!!

#lm_vocab = data_utils.CharsVocabulary(FLAGS.lm_vocab_file, FLAGS.lm_max_word_len)
#print("Loading a giant freaking external language model, its big man, reallll big")
#lm_sess, lm_model = _LoadModel(FLAGS.lm_pbtxt, FLAGS.lm_ckpt)
#print("Wheew, done loading that big model")

#def _LoadModel(gd_file, ckpt_file, sess=None):
#  """Load the model from GraphDef and Checkpoint.
#
#  Args:
#    gd_file: GraphDef proto text file.
#    ckpt_file: TensorFlow Checkpoint file.
#
#  Returns:
#    TensorFlow session and tensors dict.
#  """
#  with tf.Graph().as_default():
#    sys.stderr.write('Recovering graph.\n')
#    with tf.gfile.FastGFile(gd_file, 'r') as f:
#      s = f.read().decode()
#      gd = tf.GraphDef()
#      text_format.Merge(s, gd)
#
#    tf.logging.info('Recovering Graph %s', gd_file)
#    t = {}
#    [t['states_init'], t['lstm/lstm_0/control_dependency'],
#     t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
#     t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
#     t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
#     t['all_embs'], t['softmax_weights'], t['global_step']
#    ] = tf.import_graph_def(gd, {}, ['states_init',
#                                     'lstm/lstm_0/control_dependency:0',
#                                     'lstm/lstm_1/control_dependency:0',
#                                     'softmax_out:0',
#                                     'class_ids_out:0',
#                                     'class_weights_out:0',
#                                     'log_perplexity_out:0',
#                                     'inputs_in:0',
#                                     'targets_in:0',
#                                     'target_weights_in:0',
#                                     'char_inputs_in:0',
#                                     'all_embs_out:0',
#                                     'Reshape_3:0',
#                                     'global_step:0'], name='')
#
#    sys.stderr.write('Recovering checkpoint %s\n' % ckpt_file)
#    if not sess:
#        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) #?? extract it
#        
#    sess.run('save/restore_all', {'save/Const:0': ckpt_file})
#    sess.run(t['states_init'])
#
#  return sess, t
#
#def getNLM_Softmax(prefix_words, vocab, sess, t):
#  """Predict next words using the given prefix words.
#
#  Args:
#    prefix_words: Prefix words.
#    vocab: Vocabulary. Contains max word chard id length and converts between
#        words and ids.
#    sess and t should be the returns from _LoadModel
#  """
#  BATCH_SIZE=1
#  NUM_TIMESTEPS=1
#  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
#  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)
#
#  #sess, t = _LoadModel(FLAGS.lm_pbtxt, FLAGS.lm_ckpt)
#
#  if prefix_words.find('<S>') != 0:
#    prefix_words = '<S> ' + prefix_words
#
#  prefix = [vocab.word_to_id(w) for w in prefix_words.split()] #unknown words are given the id of unk
#  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
#  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
#  char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
#  samples = prefix[:]
#  char_ids_samples = prefix_char_ids[:]
#
#  print(samples)
#  print(vocab.max_word_length)
#
# 
#  inputs[0, 0] = samples[0]
#  char_ids_inputs[0, 0, :] = char_ids_samples[0]
#  samples = samples[1:]
#  char_ids_samples = char_ids_samples[1:]
#
#
#  print("char_ids_inputs: ", char_ids_inputs)
#  print("inputs: ", inputs)
#
#  
#  softmax = sess.run(t['softmax_out'],
#                 feed_dict={t['char_inputs_in']: char_ids_inputs,
#                            t['inputs_in']: inputs,
#                            t['targets_in']: targets,
#                            t['target_weights_in']: weights})
#
#  return softmax




    
