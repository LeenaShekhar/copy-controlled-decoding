#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
use this code to extract train/eval/test summaries from cnn/dm .bin files.
first it extrcts them into files then replaces word with count < threshold with <unks> and saves with postfix .preprocessed.

"""
import os
import sys
import glob
import struct
import random
import collections
import tensorflow as tf
from tensorflow.core.example import example_pb2

'''
use the following code to remove newlines if needed.
cat lm_train_data.txt | sed '/^\s*$/d' > lm_train_data_processed.txt
'''

# <s> and </s> are used in the data files to segment the abstracts into sentences. They don't receive vocab ids.
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
Py3 = sys.version_info[0] == 3

def example_generator(data_path, fname, single_pass=True):
  """
  Args:
    data_path:
      Path to tf.Example data files. Can include wildcards, e.g. if you have several training data chunk files train_001.bin, train_002.bin, etc, then pass data_path=train_* to access them all.
    single_pass:
      Boolean. If True, go through the dataset exactly once, generating examples in the order they appear, then return. Otherwise, generate random examples indefinitely.

  Yields:
    Deserialized tf.Example.
  """
  
  with open(fname, "w") as myfile:
    
      while True:
        filelist = glob.glob(data_path) # get the list of datafiles
        assert filelist, ('Error: Empty filelist at %s' % data_path) # check filelist isn't empty
                  
        
        if single_pass: 
            filelist = sorted(filelist)
        else:
          random.shuffle(filelist)
        
        i = 0
        for f in filelist:
          reader = open(f, 'rb')

          while True:
            print("processing: ", i)  
            len_bytes = reader.read(8)
            if not len_bytes: break # finished reading this file
            str_len = struct.unpack('q', len_bytes)[0]
            example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
            e = example_pb2.Example.FromString(example_str)
            try:
                # use this to get summaries
                abstract_text = e.features.feature['abstract'].bytes_list.value[0]
                abstract_sentences = [sent.strip() for sent in abstract2sents(abstract_text)]
                abstract = ' '.join(abstract_sentences) # string
                myfile.write("%s\n"%abstract)
                
                # use this to get articles
#                article = e.features.feature['article'].bytes_list.value[0] 
#                article=article.lower()
#                article_words = article.split()
#                article_words = article_words[:400]
#                article = ' '.join(article_words)
#                myfile.write("%s\n"%article)

            except ValueError:
                tf.logging.error('Failed to get article or abstract from example')
                continue
            
            i += 1
           
        if single_pass:
            myfile.close()
            print "example_generator completed reading all datafiles. No more data."
            break
    
    
def abstract2sents(abstract):
  """Splits abstract text from datafile into list of sentences.

  Args:
    abstract: string containing <s> and </s> tags for starts and ends of sentences

  Returns:
    sents: List of sentence strings (no tags)"""
  cur = 0
  sents = []
  while True:
    try:
      start_p = abstract.index(SENTENCE_START, cur)
      end_p = abstract.index(SENTENCE_END, start_p + 1)
      cur = end_p + len(SENTENCE_END)
      sents.append(abstract[start_p+len(SENTENCE_START):end_p])
    except ValueError as e: # no more sentences
      return sents

def ptb_raw_data(data_path="lm_data/", unk_threshold=2):
  """
  Args:
    data_path: string path to the directory where simple-examples.tgz has
      been extracted.

  Returns:
    tuple (train_data, valid_data, test_data, vocabulary)
    where each of the data objects can be passed to PTBIterator.
  """
  
  train_path = os.path.join(data_path, "lm_train_data.txt")
  valid_path = os.path.join(data_path, "lm_valid_data.txt")

  #build vocabulary
  word_to_id = build_vocab(train_path, unk_threshold=unk_threshold)

  file_to_file_with_unks(train_path, word_to_id, "lm_data/lm_train_data.txt")
  file_to_file_with_unks(valid_path, word_to_id, "lm_data/lm_valid_data.txt")
  

def read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    if Py3:
      return f.read().replace("\n", "<eos>").split()
    else:
      return f.read().decode("utf-8").replace("\n", "<eos>").split()


def get_lines(filename):
  with tf.gfile.GFile(filename, "r") as f:
      return f.readlines()

def build_vocab(filename, unk_threshold):
  
  data = read_words(filename)

  counter = collections.Counter(data)
  print("total words before: %d"%len(counter))
  counter = collections.Counter({x : counter[x] for x in counter if counter[x] >= unk_threshold })
  print("total words after: %d"%len(counter))
  
  count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0])) #(word, count)
  
  words, _ = list(zip(*count_pairs)) #words is a tuple containing all the words
  words = ("<unk>",) + words #add <unk> to words so that it can be included in the vocab
  print("total words after adding unk: %d"%len(words))
        
  word_to_id = dict(zip(words, range(len(words)))) 
  
  return word_to_id


def file_to_file_with_unks(filename, word_to_id, pfilename):

  #replace words with count < threshold with <unk>
  my_data = get_lines(filename)
  new_data = []
  for line in my_data:
      line = line.decode("utf-8")
      new_line = [word if word in word_to_id else "<unk>" for word in line.split()]
      new_data.append(u' '.join(new_line).encode('utf-8'))
  print(len(new_data))
  
  pfilename = pfilename + ".processed"
  with open(pfilename, "w") as myfile:
      for line in new_data:
          myfile.write("%s\n"%line)
  print("DONE")
  


    
#if __name__ == "__main__":
    
    #saves summaries
#    example_generator("/home/leena/Documents/thesis/pointer-gen/finished_files/chunked/train_*", "lm_data/lm_train_data.txt")
#    example_generator("/home/leena/Documents/thesis/pointer-gen/finished_files/chunked/val_*", "lm_data/lm_valid_data.txt")
#    example_generator("/home/leena/Documents/thesis/pointer-gen/finished_files/chunked/test_*", "lm_data/lm_test_data.txt")
    
    # hack to get articles
#    example_generator("/home/leena/Documents/thesis/pointer-gen/finished_files/chunked/val_*", "lm_data/lm_val_article.txt")
    
    #saves processed summaries: with <unks>
#    ptb_raw_data(data_path="lm_data/", unk_threshold=8)
        