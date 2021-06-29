from src.constants import MASK, SEP, BERT_TOKENIZER_CACHE_DIR, BERT_VERSION
import torch
import numpy as np
import time
import math
from src.fine_tuning.bert_lm import BertPred
from transformers import BertTokenizer

def tokenize_batch(batch, tokenizer):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def get_init_text(seed_sentence, max_len, tokenizer, batch_size=1):
    batch = [seed_sentence + [MASK] * max_len + [SEP] for _ in range(batch_size)]
    return tokenize_batch(batch, tokenizer)

def untokenize_batch(batch, tokenizer):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def generate_step(out, gen_idx, temperature=None, top_k=0, sample=False, return_list=True):
    """ Generate a word from from out[gen_idx]
    
    args:
        - out (torch.Tensor): tensor of logits of size batch_size x seq_len x vocab_size
        - gen_idx (int): location for which to generate for
        - top_k (int): if >0, only sample from the top k most probable words
        - sample (Bool): if True, sample from full distribution. Overridden by top_k 
    """
    logits = out[:, gen_idx]
    if temperature is not None:
        logits = logits / temperature
    if top_k > 0:
        kth_vals, kth_idx = logits.topk(top_k, dim=-1)
        dist = torch.distributions.categorical.Categorical(logits=kth_vals)
        idx = kth_idx.gather(dim=1, index=dist.sample().unsqueeze(-1)).squeeze(-1)
    elif sample:
        dist = torch.distributions.categorical.Categorical(logits=logits)
        idx = dist.sample().squeeze(-1)
    else:
        idx = torch.argmax(logits, dim=-1)
    return idx.tolist() if return_list else idx

def parallel_sequential_generation(seed_text, tokenizer, model, batch_size=1, max_len=15, top_k=0, temperature=None, max_iter=300, burnin=200,
                                   cuda=False, print_every=10, verbose=True):
    """ Generate for one random position at a timestep
    
    args:
        - burnin: during burn-in period, sample from full distribution; afterwards take argmax
    """
    seed_len = len(seed_text)
    batch = get_init_text(seed_text, max_len, batch_size)
    mask_id = tokenizer.convert_tokens_to_ids([MASK])[0]
    for ii in range(max_iter):
        kk = np.random.randint(0, max_len)
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = mask_id
        inp = torch.tensor(batch).cuda() if cuda else torch.tensor(batch)
        out = model(inp)
        topk = top_k if (ii >= burnin) else 0
        idxs = generate_step(out, gen_idx=seed_len+kk, top_k=topk, temperature=temperature, sample=(ii < burnin))
        for jj in range(batch_size):
            batch[jj][seed_len+kk] = idxs[jj]
            
        if verbose and np.mod(ii+1, print_every) == 0:
            for_print = tokenizer.convert_ids_to_tokens(batch[0])
            for_print = for_print[:seed_len+kk+1] + ['(*)'] + for_print[seed_len+kk+1:]
            print("iter", ii+1, " ".join(for_print))
            
    return untokenize_batch(batch, tokenizer)

def generate(n_samples, class_name, model, seed_text="[CLS]", batch_size=10, max_len=25, 
             sample=True, top_k=100, temperature=1.0, burnin=200, max_iter=500,
             cuda=False, print_every=1):
    # main generation function to call
    sentences = []
    n_batches = math.ceil(n_samples / batch_size)
    start_time = time.time()
    tokenizer = BertTokenizer.from_pretrained(BERT_VERSION, cache_dir=BERT_TOKENIZER_CACHE_DIR)
    for batch_n in range(n_batches):
        batch = parallel_sequential_generation(seed_text, tokenizer, model, max_len=max_len, top_k=top_k,
                                               temperature=temperature, burnin=burnin, max_iter=max_iter, 
                                               cuda=cuda, verbose=False)
        
        if (batch_n + 1) % print_every == 0:
            print("Finished batch %d in %.3fs" % (batch_n + 1, time.time() - start_time))
            start_time = time.time()
        
        sentences += batch
    return sentences