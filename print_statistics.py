#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
use this code to extract these four metrics:
    1. ROUGE
    2. METEOR
    3. REPETITION WITHIN SUMMARY
    4. OVERLAP WITH ARTICLE
    5. AVG SENTS and LEN SUMMARIES

"""

import os
import glob
import json
import pyrouge
import hashlib
import logging
import subprocess
import numpy as np
from nltk import ngrams
import tensorflow as tf
from collections import Counter
from subprocess import CalledProcessError
from sklearn.feature_extraction.text import CountVectorizer


################ ARTICLE OVERLAP ################
def get_overlap_all(summary_path, article_path, num, gold=False):
    
    if os.path.isdir(article_path):
        art_path = os.path.join(article_path, "articles/*.txt")
        art_files = sorted(glob.glob(art_path))
    else:
        art_reader = open(article_path, 'r')
        art_files = art_reader.readlines() #assuming each line is 1 a multi-sentence summary
    
    if gold:    
        sum_path = os.path.join(summary_path, "reference/*.txt") #should be decoded for all except gold labels
    else:
        sum_path = os.path.join(summary_path, "decoded/*.txt") 
        
    sum_files = sorted(glob.glob(sum_path))


    #assert len(art_files) == len(sum_files), "num articles [%d] != num summaries [%d]"%(len(art_files), len(sum_files))

#    art_files = art_files[:101]
#    sum_files = sum_files[:101]
    
    # ngram length : match_count
    match_count = dict()
    
    for idx, sum_file in enumerate(sum_files):
        
        if os.path.isdir(article_path):
            art_reader = open(art_files[idx], 'r') 
            art = art_reader.read()
        else:
            art = art_files[idx]
    
        art = art.replace("\n", " ")  
        
        sum_reader = open(sum_file, 'r')
        summ = sum_reader.read()
        summ = summ.replace("\n", " ")
        
        #print("\n[%d]"%idx)
        get_overlap(art, summ, match_count, num)
    
    print("") 
    ngram_wanted = [2, 3, 5, 8, 10, 12, 15, 18, 20, 25]
    for key, value in match_count.iteritems():
        if key in ngram_wanted:
            #print("%d-gram avg match: %.3f"%(key, value/float(len(sum_files))))    
            print("%.3f"%(value/float(len(sum_files))))

#ngram_wanted = [2, 3, 5, 8, 10, 12, 15, 18, 20, 25]
AVG_OVERLAP = [94.112, 87.054, 75.152, 59.769, 50.363, 41.681, 30.951, 22.996, 18.916, 11.902]
# get_overlap('to ngramize it i', 'this is a foo bar sentences and i want to ngramize it', num=5)    
def get_overlap(article, summary, match_count=None, num=5):
    # get all n-grams from n to 1
    
    if match_count is None:
        match_count = dict()
        
    for n in range(num): #goes from 0 to num-1
        # create ngrams wih n
        art_split = article.split()
        ngart = list(ngrams(art_split[:400], n+1))
        ngsum = list(ngrams(summary.split(), n+1))
        # 1 if matches else 0
        ngmatch = [1 if x in ngart else 0 for x in ngsum]
        
        '''
        if n+1 >= 30:
            ngmatch_str = [x for x in ngsum if x in ngart]
            print("ngmatch_str for %d-gram: %s"%(n+1, ngmatch_str))
        '''
        
        # dict steps
        if n+1 not in match_count:
            match_count[n+1] = 0 #initialize
                       
        #match_count[n+1] += sum(ngmatch) #update sum
        
        tot_len = float(len(ngsum))
        #print("# %d-grams in summary: %.3f"%(n+1, tot_len))
        if tot_len == 0: #handle zero div error
            tot_len = 1
        match_count[n+1] += 100 * (sum(ngmatch)/tot_len) #update fraction
#        if n+1 in ngram_wanted:
#            print("# %d-grams: %.3f"%(n+1, 100 * (sum(ngmatch)/tot_len)) )

 
################ SUMMARY REPETITION ################
vectorizer = CountVectorizer()

def count_repeated_sentences(base_path, gold=False):
  '''
  read all files from base_path and finds repetition in a file: exact and approximate
  '''  
  if gold:
      hyp_path = os.path.join(base_path, "reference/*.txt") ##should be decoded for all except gold labels
  else:
      hyp_path = os.path.join(base_path, "decoded/*.txt") 
      
  hypsfilelist = sorted(glob.glob(hyp_path))
  
#  hypsfilelist = hypsfilelist[:101]
      
  corpus_repeat_fnames = []
  corpus_bow_repeat_fnames = []
  
  corpus_len_dist = dict()
  corpus_repeat_len_dist = dict()
  corpus_repeat_indices = []
  corpus_bow_repeat_indices = []
  
  for idx, fname in enumerate(hypsfilelist):
    doc = open(fname, 'r')
    sentences = doc.readlines()

    sentences = [sentence.strip() for sentence in sentences] 
    count_exact_repeated_sentences(sentences, corpus_repeat_fnames, fname, corpus_repeat_len_dist, corpus_repeat_indices)
    #count_bow_repeated_sentences(sentences, corpus_bow_repeat_fnames, fname, corpus_bow_repeat_indices)

    for sentence in sentences:
              corpus_len_dist[len(sentence)] = 0 if len(sentence) not in corpus_len_dist else corpus_len_dist[len(sentence)] + 1

  #print('\navg num summaries with atleast 1 repetition: {%s}/{%s}' %(len(corpus_repeat_fnames),len(hypsfilelist)))
  print('\navg repetition: %.3f' %( 100 * (len(corpus_repeat_fnames)/float(len(hypsfilelist)))) )
  #print('number of summaries with .9 repeated senences: {%s}/{%s}' %(len(corpus_bow_repeat_fnames),len(hypsfilelist)))

  files_with_exact_matches = corpus_repeat_fnames
  files_with_approx_matches = sorted(set(corpus_bow_repeat_fnames) - set(corpus_repeat_fnames))
  #print('repetition in files: %s'%corpus_repeat_fnames)

  return corpus_repeat_indices, corpus_bow_repeat_indices, files_with_exact_matches, files_with_approx_matches
  
  
def count_exact_repeated_sentences(sentences, corpus_repeat_fnames, fname, corpus_repeat_len_dist, corpus_repeat_indices):
   '''
   finds exact repetition by comparing hash of strings
   '''
   hashes = [hashlib.md5(sentence).hexdigest() for sentence in sentences]
   # implies duplicate elements in the list
   if len(hashes) > len(set(hashes)):
       corpus_repeat_fnames.append(fname.split('/')[-1])
   # filtered hashes for repetition
   repeated_hash = [k for k, v in Counter(hashes).items() if v > 1]
   repeat_indices = []
   for hsh in repeated_hash:
       # indx corresponds to the sentence id
       indices = [i for i, x in enumerate(hashes) if x == hsh]
       indx = indices[0]
       corpus_repeat_len_dist[len(sentences[indx])] = 0 if len(sentences[indx]) not in corpus_repeat_len_dist else corpus_repeat_len_dist[len(sentences[indx])] + 1
       for r_idx in indices:                       
           repeat_indx = len(" ".join(s for s in sentences[:r_idx]).split(" ")) #start from 0 so no need to add +1
           repeat_indices.append(repeat_indx)
   corpus_repeat_indices.append(repeat_indices)
                              
# TODO: this need some fixing        
def count_bow_repeated_sentences(sentences, corpus_bow_repeat_fnames, fname, corpus_bow_repeat_indices):
    '''
    finds words a matching indices encoded using BoW by using logical AND condition
    '''
    repeat = False
    repeat_indices = []
    indices = []
    X = vectorizer.fit_transform(sentences)
    X = X.toarray()
    X = X == 1 # boolean to 1
    for idx1, row1 in enumerate(X[:-1]):
        for idx2, row2 in enumerate(X[idx1+1:]):
            if np.sum(row1 & row2)/(np.sum(row1)+ 10^-7) >= 0.9:
                repeat = True
                repeat_indices.extend([idx1, idx1+1]) #use this list to only keep track of approx matches so subtract extact ones aa there be some idx issues
                
    for r_idx in repeat_indices:                       
        repeat_indx = len(" ".join(s for s in sentences[:r_idx]).split(" ")) #start from 0 so no need to add +1
        indices.append(repeat_indx)           
        
    if repeat:
        corpus_bow_repeat_fnames.append(fname.split('/')[-1]) 
        
    corpus_bow_repeat_indices.append(indices)    


################ AVG LEN + SENTS ################
def get_avg_stats(base_path, gold=False):
    
    if gold:
        hyp_path = os.path.join(base_path, "reference/*.txt") ##should be decoded for all except gold labels
    else:
        hyp_path = os.path.join(base_path, "decoded/*.txt")     
        
    hypsfilelist = sorted(glob.glob(hyp_path))
    
    #hypsfilelist = hypsfilelist[:5]
    
    total_nsentence = 0
    total_words = 0
    
    for f in hypsfilelist:
        reader = open(f, 'r')
        hyp = reader.read()
        hyp = hyp.replace("\n", " ")
        
        total_nsentence += len(hyp.strip().split(".")) #sentences are seperated by "."
        total_words += len(hyp.strip().split()) # words are eperated by " "
        
        #print("hyp: ", hyp.strip())                  
        #print("nsentence: ", len(hyp.strip().split(".")))
        #print("words: ", len(hyp.strip().split()))
        
    avg_nsentence, avg_length = total_nsentence/float(len(hypsfilelist)), total_words/float(len(hypsfilelist))

    print("\navg num sentences per summary: %.3f"%avg_nsentence)
    print("avg length of a summary: %.3f"%avg_length)
    

################ METEOR ################
def evaluate_meteor(base_path, exact=False):

  ref_path = os.path.join(base_path, "reference/*.txt")
  hyp_path = os.path.join(base_path, "decoded/*.txt")
  
  refsfilelist = sorted(glob.glob(ref_path))
  hypsfilelist = sorted(glob.glob(hyp_path))
  
#  refsfilelist = refsfilelist[:101]
#  hypsfilelist = hypsfilelist[:101]
  
  refs = []
  hyps = []
  
  for f in refsfilelist:
      reader = open(f, 'r')
      ref = reader.read()
      ref = ref.replace("\n", " ")
      refs.append(ref)

  ref_filename = os.path.join(base_path, "_temp_refs") #temp file
  with open(ref_filename, "w") as myfile:
      for ref in refs:
          myfile.write("%s\n"%ref)

  for f in hypsfilelist:
      reader = open(f, 'r')
      hyp = reader.read()
      hyp = hyp.replace("\n", " ")
      hyps.append(hyp)

  hyp_filename = os.path.join(base_path, "_temp_hyps") #temp file
  with open(hyp_filename, "w") as myfile:
      for hyp in hyps:
          myfile.write("%s\n"%hyp)
  
  assert len(refs) == len(hyps), "length of references and hypothesis are different."

  # exact   
  if exact:     
      cmd = 'java -Xmx2G -jar meteor-1.5/meteor-1.5.jar "%s" "%s" -norm -m exact' % (hyp_filename, ref_filename)  
  # exact + stem + syn + para          
  else:
      cmd = 'java -Xmx2G -jar meteor-1.5/meteor-1.5.jar "%s" "%s" -norm' % (hyp_filename, ref_filename)


  
  try:
    output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
  except CalledProcessError as e:
    output = e.output

  # remove the temp files
  if os.path.exists(ref_filename):
      os.remove(ref_filename)

  if os.path.exists(hyp_filename):
      os.remove(hyp_filename)

  # process the output str
  final_score = None
  result_str = ""
  wanted = ["Modules"]#, "Precision", "Recall", "f1", "fMean", "Fragmentation penalty", "Test words", "Reference words", "Chunks"]
  
  for _l in output.split('\n'):
    l = _l.strip()
    if len(l) == 0:
      continue

    tokens = l.split(":")
    if len(tokens) != 2:
      continue

    if tokens[0] == "Final score":
      final_score = float(tokens[1].strip())*100.0
      result_str += "%s"%l                 
      break
    elif tokens[0] in wanted and not tokens[0].startswith("Segment"):
        result_str += "%s\n"%l
    
  print("\n METEOR SCORES: \n%s"%result_str) 


################ ROUGE ################
def rouge_eval(base_path):
  """Evaluate the files in ref_dir and dec_dir with pyrouge, returning results_dict"""
  
  ref_dir = os.path.join(base_path, "reference")
  dec_dir = os.path.join(base_path, "decoded")
  
  r = pyrouge.Rouge155()
  r.model_filename_pattern = '#ID#_reference.txt'
  r.system_filename_pattern = '(\d+)_decoded.txt'
  r.model_dir = ref_dir
  r.system_dir = dec_dir
  logging.getLogger('global').setLevel(logging.WARNING) # silence pyrouge logging
  rouge_results = r.convert_and_evaluate()
  
  results_dict = r.output_to_dict(rouge_results)
  rouge_log(results_dict, base_path)



def rouge_log(results_dict, dir_to_write):
  """Log ROUGE results to screen and write to file.

  Args:
    results_dict: the dictionary returned by pyrouge
    dir_to_write: the directory where we will write the results to"""
  log_str = ""
  for x in ["1","2","l"]:
    log_str += "\nROUGE-%s:\n" % x
    for y in ["f_score", "recall", "precision"]:
      key = "rouge_%s_%s" % (x,y)
      key_cb = key + "_cb"
      key_ce = key + "_ce"
      val = results_dict[key]
      val_cb = results_dict[key_cb]
      val_ce = results_dict[key_ce]
      log_str += "%s: %.4f with confidence interval (%.4f, %.4f)\n" % (key, val, val_cb, val_ce)
  print(log_str) # log to screen
  results_file = os.path.join(dir_to_write, "ROUGE_results.txt") 
  tf.logging.info("Writing final ROUGE results to %s...", results_file)
  with open(results_file, "w") as f:
    f.write(log_str)



def extract_from_json(base_path):

    attn_vis_path = os.path.join(base_path, "attn_vis")
    num  = len(glob.glob(attn_vis_path+"/*.json"))
    
    idx = 0
    total_log_prob = 0.0
    total_sum_log_prob = 0.0
    total_pgen = 0.0
    for idx in range(num): #goes till num-1
            json_data = json.load(open(attn_vis_path+"/%06d_attn_vis_data.json"%idx))
            try:
                total_log_prob += json_data['log_prob'] #one value
            except:
                total_log_prob += sum(json_data['log_probs']) #list
                
            total_sum_log_prob += json_data["avg_log_prob"] #one value
            total_pgen += json_data["avg_pgen"] #one value
            
    print("avg_log_prob: %.3f"%(total_log_prob/num))
    print("avg_sum_log_prob: %.3f"%(total_sum_log_prob/num)) 
    print("avg_pgen: %.3f"%(total_pgen/num))
        
    

def get_all_stats(base_path,  article_path, gold=False, scores=False, exact=False, baseline=False):
    
    if not gold and scores:
        rouge_eval(base_path)
        evaluate_meteor(base_path)
        evaluate_meteor(base_path, exact=True)
        #extract_from_json(base_path)
        
    if baseline:
        rouge_eval(base_path)
        evaluate_meteor(base_path)
        evaluate_meteor(base_path, exact=True)
        
    #ordering of samples does not matter in the following stats
    get_avg_stats(base_path, gold=gold)
    count_repeated_sentences(base_path, gold=gold)
    get_overlap_all(base_path, article_path, num=30, gold=gold) 

    
    

# this points to all the merged test articles
article_path = "/home/leena/Documents/thesis/pointer-gen/test_output/all_test_article_truncated.txt"

# this points to all the merged validation articles
val_article_path = "/home/leena/Documents/thesis/pointer-gen/test_output/all_val_article_truncated.txt"

###############################################
# Below Experiments were run
###############################################



#### these would be reported on the test dataset ####

#print("\n***Baseline***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/test_output/temp/", "/home/leena/Documents/thesis/pointer-gen/test_output/temp/", baseline=True)

#print("\n***Reference***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/test_output/", article_path, gold=True, scores=True)

#print("\n***Pre Coverage***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/dont_delete_8000_decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-237470/all_decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-237470/", article_path, scores=True)
#
#print("\n***Coverage***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/cov-org-dont-delete/all_decode_test_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/", article_path, scores=True)

# OUR PROPOSED METHOD OUTPUTS [Below 2 didn't work as well so we moved to the increasing penalty one]

#print("\n***Penalty without coverage***\n")
#get_all_stats("/home/leena/Downloads/decode_hinge_4_04_precov_test_full/decode_test_400maxenc_4beam_35mindec_120maxdec_final_test_0_6000penalty-ckpt-237470/", article_path, scores=True)
#
#print("\n***Penalty with coverage***\n")
#get_all_stats("/home/leena/Downloads/decode_hinge_5_04_cov_test_full/decode_test_400maxenc_4beam_35mindec_120maxdec_final_test_cov_0_6000penalty-ckpt-238410/", article_path, scores=True)


##### increasing penalty [Final proposed method]

#print("\n*** Inc Penalty without coverage***\n")
#get_all_stats("/home/leena/Downloads/DECODE_NONCOV_FINAL_1/decode_test_400maxenc_4beam_35mindec_120maxdec_NONCOV_NONLOG_FINAL_0_1400penalty-ckpt-237470/", article_path, scores=True)

#print("\n***Inc Penalty with coverage***\n")
#get_all_stats("/home/leena/Downloads/DECODED_COV_FINAL_NONLOG/decode_test_400maxenc_4beam_35mindec_120maxdec_COV_NONLOG_FINAL_0_3000penalty-ckpt-238410/", article_path, scores=True)




###################################################
#### these would be reported on the val dataset ####



#### pre cov param search experments ####

#print("\n***pre cov***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_val_ckpt-237470/", val_article_path, scores=True)

#print("\n***100-44***\n")

#print("\n***CE 1 40***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_val_ce_ckpt-237470/", val_article_path, scores=True)
#print("\n***CE 3 45***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_kenlm-penalty-ckpt-237470/", val_article_path, scores=True)
#print("\n***CE 2 45***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_pen_2_target_45_ckpt-237470/", val_article_path, scores=True)
#print("\n***CE 2.5 53***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_kenlm-penalty-ckpt-237470/", val_article_path, scores=True)


#print("\n***40 40***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_40_04penalty-ckpt-237470/", val_article_path, scores=True)
#print("\n***60 40***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_60_04penalty-ckpt-237470/", val_article_path, scores=True)

#print("\n***Hinge margin 5 40 ***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_margin_5_target_40ckpt-237470/", val_article_path, scores=True)
#print("\n***Hinge margin 7 40 ***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_margin_7_target_40ckpt-237470/", val_article_path, scores=True)
#print("\n***Hinge margin 6 40 ***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/pre-coverage/decode_val_400maxenc_4beam_35mindec_120maxdec_pre_cov_margin_6_target_40ckpt-237470/", val_article_path, scores=True)
#print("\n***Hinge margin 4 40 ***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_hinge_4_04penalty-ckpt-237470/", val_article_path, scores=True)

#print("\n***Non Log Inc***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_non_cov_increasing_nonlog_45_04_01penalty-ckpt-237470/", val_article_path, scores=True)



#### cov param search experiments ####

#print("\n***cov***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_ckpt-238410/", val_article_path, scores=True)

#print("\n***100-44***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_trial-ckpt-238410/", val_article_path, scores=True)
#print("\n***80-44***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_trial-80-44-ckpt-238410/", val_article_path, scores=True)
#print("\n***80-53***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_trial-80-53-ckpt-238410/", val_article_path, scores=True)
#print("\n***70-50***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_trial-70-50-ckpt-238410/", val_article_path, scores=True)
#print("\n***80-40***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_trial-80-40-ckpt-238410/", val_article_path, scores=True)
#print("\n***70 40***\n")
#get_all_stats("/home/leena/Documents/thesis/pointer-gen/log/cov/decode_val_400maxenc_4beam_35mindec_120maxdec_cov_penalty_70_target_40_ckpt-238410/", val_article_path, scores=True)

#print("\n***Hinge 5 .4***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_hinge_5_04penalty-ckpt-238410/", val_article_path, scores=True)
#print("\n***Hinge 7 .4***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_hinge_7_04penalty-ckpt-238410/", val_article_path, scores=True)
#print("\n***Hinge 4 .4***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_hinge_4_04penalty-ckpt-238410/", val_article_path, scores=True)

#print("\n***Log Inc***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_increasing5_04penalty-ckpt-238410/", val_article_path, scores=True)
#print("\n***Non Log Inc***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_increasing_nonlog_5_04penalty-ckpt-238410/", val_article_path, scores=True)

#print("\n***Non Log Inc 5 04 1***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_increasing_nonlog_5_04_01penalty-ckpt-238410/", val_article_path, scores=True)
#print("\n***Log Inc 5 04 5***\n")
#get_all_stats("/home/leena/Downloads/decode_val_400maxenc_4beam_35mindec_120maxdec_increasing_log_5_04_05penalty-ckpt-238410/", val_article_path, scores=True)

