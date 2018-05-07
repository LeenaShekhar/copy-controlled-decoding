# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
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

"""This file contains code to run beam search decoding"""

import tensorflow as tf
import numpy as np
import data
from google.protobuf import text_format
import sys

# Use external LM to see if performance increases.
from my_lm_1b_eval import get_RNNLM_softmax, get_KenLM_Softmax


FLAGS = tf.app.flags.FLAGS

BETA_LM = 0.3

AVERAGE_PCOPY_THRESHOLD=0.4 # this is m* in the paper

# MSE penalty
ALPHA_PEN = 70.0

# Cross entropy penalty 
CE_PEN = 2.0

# Margin Penalty
ETA = 0.5


class Hypothesis(object):
  """Class to represent a hypothesis during beam search. Holds all the information needed for the hypothesis."""

  def __init__(self, tokens, log_probs, state, attn_dists, p_gens, coverage, per_sentence_states, per_sentence_cstates, per_sentence_igates, per_sentence_fgates, per_sentence_ogates, per_sentence_ccands, per_sentence_cstateinps, per_sentence_cstatefgts, per_sentence_canduws, unnormalized_attn_dists, average_pgen=0):
    """Hypothesis constructor.

    Args:
      tokens: List of integers. The ids of the tokens that form the summary so far.
      log_probs: List, same length as tokens, of floats, giving the log probabilities of the tokens so far.
      state: Current state of the decoder, a LSTMStateTuple.
      attn_dists: List, same length as tokens, of numpy arrays with shape (attn_length). These are the attention distributions so far.
      p_gens: List, same length as tokens, of floats, or None if not using pointer-generator model. The values of the generation probability so far.
      coverage: Numpy array of shape (attn_length), or None if not using coverage. The current coverage vector.
    """
    self.tokens = tokens
    self.log_probs = log_probs
    self.state = state
    self.attn_dists = attn_dists
    self.p_gens = p_gens
    self.coverage = coverage

    self.average_pgen = average_pgen
    
    self.per_sentence_states = per_sentence_states
    self.per_sentence_cstates = per_sentence_cstates
    self.per_sentence_igates = per_sentence_igates
    self.per_sentence_fgates = per_sentence_fgates
    self.per_sentence_ogates = per_sentence_ogates
    
    self.per_sentence_ccands = per_sentence_ccands
    self.per_sentence_cstateinps = per_sentence_cstateinps
    self.per_sentence_cstatefgts = per_sentence_cstatefgts
    self.per_sentence_canduws = per_sentence_canduws
    
    self.unnormalized_attn_dists = unnormalized_attn_dists
    


  def extend(self, token, log_prob, state, attn_dist, p_gen, coverage, per_sentence_state, per_sentence_cstate, per_sentence_igate, per_sentence_fgate, per_sentence_ogate, per_sentence_ccand, per_sentence_cstateinp, per_sentence_cstatefgt, per_sentence_canduw, unnormalized_attn_dist, avg_pgen):
    """Return a NEW hypothesis, extended with the information from the latest step of beam search.

    Args:
      token: Integer. Latest token produced by beam search.
      log_prob: Float. Log prob of the latest token.
      state: Current decoder state, a LSTMStateTuple.
      attn_dist: Attention distribution from latest step. Numpy array shape (attn_length).
      p_gen: Generation probability on latest step. Float.
      coverage: Latest coverage vector. Numpy array shape (attn_length), or None if not using coverage.
    Returns:
      New Hypothesis for next step.
    """
    return Hypothesis(tokens = self.tokens + [token],
                      log_probs = self.log_probs + [log_prob],
                      state = state,
                      attn_dists = self.attn_dists + [attn_dist],
                      p_gens = self.p_gens + [p_gen],
                      coverage = coverage,
                      per_sentence_states = self.per_sentence_states + [per_sentence_state],
                      per_sentence_cstates = self.per_sentence_cstates + [per_sentence_cstate],                                                 
                      per_sentence_igates = self.per_sentence_igates + [per_sentence_igate],
                      per_sentence_fgates = self.per_sentence_fgates + [per_sentence_fgate],
                      per_sentence_ogates = self.per_sentence_ogates + [per_sentence_ogate],
                      per_sentence_ccands = self.per_sentence_ccands + [per_sentence_ccand],
                      per_sentence_cstateinps = self.per_sentence_cstateinps + [per_sentence_cstateinp],
                      per_sentence_cstatefgts = self.per_sentence_cstatefgts + [per_sentence_cstatefgt],
                      per_sentence_canduws = self.per_sentence_canduws + [per_sentence_canduw],
                      unnormalized_attn_dists = self.unnormalized_attn_dists + [unnormalized_attn_dist], average_pgen=avg_pgen)

  @property
  def latest_token(self):
    return self.tokens[-1]

  @property
  def log_prob(self):
    # the log probability of the hypothesis so far is the sum of the log probabilities of the tokens so far
    return sum(self.log_probs)

  @property
  def avg_log_prob(self):
    # normalize log probability by number of tokens (otherwise longer sequences always have lower probability)
    return self.log_prob / len(self.tokens)


def run_beam_search(sess, model, vocab, batch, counter=0, lm_model=None, lm_word2idx=None, lm_idx2word=None):
  """Performs beam search decoding on the given example.

  Args:
    sess: a tf.Session
    model: a seq2seq model
    vocab: Vocabulary object
    batch: Batch object that is the same example repeated across the batch

  Returns:
    best_hyp: Hypothesis object; the best hypothesis found by beam search.
  """
  # Run the encoder to get the encoder hidden states and decoder initial state
  enc_states, dec_in_state = model.run_encoder(sess, batch)
  # dec_in_state is a LSTMStateTuple
  # enc_states has shape [batch_size, <=max_enc_steps, 2*hidden_dim].

  # Initialize beam_size-many hyptheses
  hyps = [Hypothesis(tokens=[vocab.word2id(data.START_DECODING)],
                     log_probs=[0.0],
                     state=dec_in_state,
                     attn_dists=[],
                     p_gens=[],
                     coverage=np.zeros([batch.enc_batch.shape[1]]), # zero vector of length attention_length
                     per_sentence_states=[],
                     per_sentence_cstates=[],                     
                     per_sentence_igates=[],
                     per_sentence_fgates=[],
                     per_sentence_ogates=[],
                     per_sentence_ccands=[],
                     per_sentence_cstateinps=[],
                     per_sentence_cstatefgts=[],
                     per_sentence_canduws=[],
                     unnormalized_attn_dists=[]) for _ in xrange(FLAGS.beam_size)]
    
  results = [] # this will contain finished hypotheses (those that have emitted the [STOP] token)

###################################################################################################

  #get the needed vocab words
  #This probably will NOT give the correct thing, as it assumes the ordering of words corresponds with the ordering of keys
  words = sorted(vocab._id_to_word.items(), key=lambda x: x[0])
  words = [x[1] for x in words]
  
  if FLAGS.use_oov_prob:
      print("adding OOV to words")
      words = words + batch.art_oovs[0]  
      
#  print(words[:5])

  if FLAGS.use_external_lm and FLAGS.use_lm_outside:
      print("using External LM outside the graph.")
      
  if FLAGS.use_nlm:
      print("using Neural LM outside the graph")
      vocab_converter = data.VocabConverter(vocab, lm_word2idx)
  
  
###################################################################################################
  steps = 0
  
  #Leena: random sampling code
  random_bos_index = [1] * FLAGS.beam_size if FLAGS.do_ablation else None #Leena: initialize BOS index
  replaced_with_random = [0] * FLAGS.beam_size if FLAGS.do_ablation else None # this keeps track of how many times random state has been added
                                                  

#  print("OOV: %s and %s", type(batch.art_oovs[0]), batch.art_oovs[0]) #list of OOV words
      
      
  while steps < FLAGS.max_dec_steps and len(results) < FLAGS.beam_size: 
    latest_tokens = [h.latest_token for h in hyps] # latest token produced by each hypothesis

    latest_tokens = [t if t in xrange(vocab.size()) else vocab.word2id(data.UNKNOWN_TOKEN) for t in latest_tokens] # change any in-article temporary OOV ids to [UNK] id, so that we can lookup word embeddings
    states = [h.state for h in hyps] # list of current decoder states of the hypotheses
    prev_coverage = [h.coverage for h in hyps] # list of coverage vectors (or None)
    # Run one step of the decoder to get the new info
    
    # this keeps track of whether period has been produced or not in a beam
    bos_info = [True if x == vocab.word2id(data.PERIOD) else False for x in latest_tokens] if FLAGS.do_ablation else None
    if bos_info is not None: print(bos_info)

###############################
    '''
    calling the LM model here to get softmax values
    '''

    word_probs = []
    
    # this is a numpy array of size 1 * 80k
    for beam_number, h in enumerate(hyps):
        tokens = h.tokens
        tokens = [int(t) for t in h.tokens[1:]] #indices
        tokens = data.outputids2words(tokens, vocab, batch.art_oovs[0])#words
        tokens = " ".join(tokens) #string

        if FLAGS.print_info:
            print("Beam Number: " + str(beam_number) + " " + tokens)
       
        if FLAGS.use_external_lm and h.latest_token != vocab.word2id(data.STOP_DECODING): #DOESNT HELP: to avoid long summaries
            # not normalized
            if FLAGS.use_nlm:    
                lm_softmax = get_RNNLM_softmax(lm_model, tokens, lm_word2idx, lm_idx2word, cuda=FLAGS.cuda) #use probs of unk for oov words
            # noormalized    
            else:
                lm_softmax = get_KenLM_Softmax(tokens, words, lm_model) #pass oov words to get their probs

        else:
            # needed because in feed cannot pass None as it breaks
            lm_softmax = np.zeros([vocab.size()], dtype=np.float)
        

        if FLAGS.use_nlm:
            # even when oov is set we add 0.0 probs so doest matter in nlm
            probs = vocab_converter.convertSoftmax(lm_softmax, oov_len=len(batch.art_oovs[0]) if FLAGS.use_oov_prob else 0) 
        else:
            probs = lm_softmax 
            
        word_probs.append(probs)


    if FLAGS.use_oov_prob:
        assert len(word_probs) == FLAGS.beam_size and len(word_probs[0]) == vocab.size() + len(batch.art_oovs[0]), "size check went wrong" 
    else:
        assert len(word_probs) == FLAGS.beam_size and len(word_probs[0]) == vocab.size(), "size check went wrong"    
    

    word_probs = np.array(word_probs) #beam_Size * vocab_size
    
      

###############################
    (topk_ids, topk_log_probs, new_states, attn_dists, p_gens, new_coverage, per_sentence_states, per_sentence_cstates, per_sentence_igates, per_sentence_fgates, per_sentence_ogates, per_sentence_ccands, per_sentence_cstateinps, per_sentence_cstatefgts, per_sentence_canduws, unnormalized_attn_dists) = model.decode_onestep(sess=sess,
                        batch=batch,
                        latest_tokens=latest_tokens,
                        enc_states=enc_states,
                        dec_init_states=states,
                        prev_coverage=prev_coverage, bos_info = bos_info, counter=counter, random_bos_index=random_bos_index, replaced_with_random=replaced_with_random, word_probs=word_probs)

#######################################
    # integrating outside the graph with KENLM
    
    if FLAGS.use_external_lm and FLAGS.use_lm_outside:

        lm_log_probs = []
        for i in xrange(FLAGS.beam_size):
            #if [STOP] in top k don't add the LM probs
            if vocab.word2id(data.STOP_DECODING) not in topk_ids[i, :5]:
                # topk_ids could be more than vocab_size: could have oov words
                log_probs = np.log(np.take(word_probs[i], topk_ids[i]))
            else:
                print("[STOP] in beam.")
                log_probs = np.zeros(topk_ids[i].shape)
            lm_log_probs.append(log_probs)
        lm_log_probs = np.array(lm_log_probs)        
                                      
        assert topk_log_probs.shape[0] == lm_log_probs.shape[0] and topk_log_probs.shape[1] == lm_log_probs.shape[1], "problem in LM probs outside the graph"         
                                                         
#        topk_log_probs = np.add( (1- BETA_LM) * topk_log_probs, BETA_LM * lm_log_probs) 
        topk_log_probs = np.log(np.add( (1- BETA_LM) * np.exp(topk_log_probs), BETA_LM * np.exp(lm_log_probs)))
    
        
    #######################################
    
    # Extend each hypothesis and collect them all in all_hyps
    all_hyps = []
    num_orig_hyps = 1 if steps == 0 else len(hyps) # On the first step, we only had one original hypothesis (the initial hypothesis). On subsequent steps, all original hypotheses are distinct.
    for i in xrange(num_orig_hyps):
      h, new_state, attn_dist, p_gen, new_coverage_i, per_sentence_state,  per_sentence_cstate, per_sentence_igate, per_sentence_fgate, per_sentence_ogate, per_sentence_ccand, per_sentence_cstateinp,  per_sentence_cstatefgt, per_sentence_canduw, unnormalized_attn_dist = hyps[i], new_states[i], attn_dists[i], p_gens[i], new_coverage[i], per_sentence_states[i], per_sentence_cstates[i], per_sentence_igates[i], per_sentence_fgates[i], per_sentence_ogates[i], per_sentence_ccands[i], per_sentence_cstateinps[i], per_sentence_cstatefgts[i], per_sentence_canduws[i], unnormalized_attn_dists[i]# take the ith hypothesis and new decoder state info
      
      #Leena: moved this code out of if to get avg score                                                                                                                                                                                                                                                                            
      iterations = steps + 1 #since steps isn't updated till later
      h.average_pgen = h.average_pgen*(float(iterations-1)/iterations) + (p_gen[0]/iterations)
      print("Average Pgen: " + str(h.average_pgen))      
      
      if FLAGS.use_penalty:
          print("Using Penalty")
          if FLAGS.use_ce:
              print("CE Penalty")
              penalty = - CE_PEN * (AVERAGE_PCOPY_THRESHOLD * np.log(h.average_pgen) + (1 - AVERAGE_PCOPY_THRESHOLD) * np.log(1 - h.average_pgen))
          elif FLAGS.use_margin:
              print("Margin Penalty")
              penalty = steps * ETA * max(0.0, (AVERAGE_PCOPY_THRESHOLD - h.average_pgen)) # THIS IS THE ONE FINALLY USED IN THE PAPER
          else:
              print("MSE Penalty")
              penalty = ALPHA_PEN*(h.average_pgen - AVERAGE_PCOPY_THRESHOLD)**2
      else:
          penalty = 0

      for j in xrange(FLAGS.beam_size * 5):  # for each of the top 2*beam_size hyps:
        # Extend the ith hypothesis with the jth option
        new_hyp = h.extend(token=topk_ids[i, j],
#                           log_prob=topk_log_probs[i, j],
                           log_prob=topk_log_probs[i, j] - penalty,
                           state=new_state,
                           attn_dist=attn_dist,
                           p_gen=p_gen,
                           coverage=new_coverage_i,
                           per_sentence_state=per_sentence_state,
                           per_sentence_cstate=per_sentence_cstate,
                           per_sentence_igate=per_sentence_igate,
                           per_sentence_fgate=per_sentence_fgate,
                           per_sentence_ogate=per_sentence_ogate,
                           per_sentence_ccand=per_sentence_ccand,
                           per_sentence_cstateinp=per_sentence_cstateinp,
                           per_sentence_cstatefgt=per_sentence_cstatefgt,
                           per_sentence_canduw=per_sentence_canduw,
                           unnormalized_attn_dist=unnormalized_attn_dist, avg_pgen=h.average_pgen)
        all_hyps.append(new_hyp)

    # Filter and collect any hypotheses that have produced the end token.
    hyps = [] # will contain hypotheses for the next step
#    print("[STOP] vocab id: ", vocab.word2id(data.STOP_DECODING))
#    print("\n")
    for h in sort_hyps(all_hyps): # in order of most likely h
      if h.latest_token == vocab.word2id(data.STOP_DECODING): # if stop token is reached...
        # If this hypothesis is sufficiently long, put in results. Otherwise discard.
        if steps >= FLAGS.min_dec_steps:
          results.append(h)
      else: # hasn't reached stop token, so continue to extend this hypothesis
          if FLAGS.no_overlap:
            if not overlapping(hyps, h) or steps < 1:
              hyps.append(h)
          else:
            hyps.append(h)

      if len(hyps) == FLAGS.beam_size or len(results) == FLAGS.beam_size:
        # Once we've collected beam_size-many hypotheses for the next step, or beam_size-many complete hypotheses, stop.
        break

    steps += 1
    
#    random_bos_index = [random_bos_index[i]+1 if x else random_bos_index[i] for i, x in enumerate(bos_info)] #beams for which period has been seen increase the count
#    print("random_bos_index: ", random_bos_index)
#    replaced_with_random = [replaced_with_random[i]+1 if x else replaced_with_random[i] for i, x in enumerate(bos_info)]
###################################################################################################


  # At this point, either we've got beam_size results, or we've reached maximum decoder steps

  if len(results)==0: # if we don't have any complete results, add all current hypotheses (incomplete summaries) to results
    results = hyps

  if FLAGS.print_info:
      for idx, h in enumerate(results):
          tokens = h.tokens
          tokens = [int(t) for t in h.tokens[1:]] #indices
          tokens = data.outputids2words(tokens, vocab, batch.art_oovs[0])#words
          tokens = " ".join(tokens) #string      
          print("%d: %s"%(idx, tokens))
  # Sort hypotheses by average log probability
  hyps_sorted = sort_hyps(results)

  # Return the hypothesis with highest average log prob
  return hyps_sorted#hyps_sorted[0]

def overlapping(hyps, candidate, hits=0, max_hits=2):
    print([h.tokens for h in hyps])
    print(candidate.tokens)
    if hyps == []:
        return False
    elif hyps[0].tokens[:len(hyps[0].tokens)-1] == candidate.tokens[:len(candidate.tokens)-1] and hyps[0].tokens[:len(hyps[0].tokens)-1] != []:
        new_hits = hits + 1
        if new_hits >= max_hits: #the candidate matches with at least max_hits in hyps
            return True
        else: #need to see if their are more matching hypothesis
            return overlapping(hyps[1:], candidate, new_hits)
    else:
        return overlapping(hyps[1:], candidate, hits)

def sort_hyps(hyps):
  """Return a list of Hypothesis objects, sorted by descending average log probability"""
  return sorted(hyps, key=lambda h: h.avg_log_prob, reverse=True)
